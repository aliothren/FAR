import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict
from timm.models import create_model
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from modeling import config
from modeling import models_act
from modeling import architectures
from modeling.data import load_dataset

FIXED_IMG_PAIRS_PATH = ""


class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.samples = base_dataset.samples

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        img, target = self.base[index]
        return img, target, index


def get_mamba_cls_scores(model, imgs):
    """
    Interface/return matches get_attn_cls_scores:
    Returns:
        scores_per_layer: list length L
            each tensor shape [B, num_patches] on CPU (excluding CLS)
    """
    model.train()
    m = model.module if hasattr(model, "module") else model

    L = len(m.blocks)
    B = imgs.shape[0]

    inputs = [None] * L
    hooks = []

    def hook_fn(module, input, output, blk_idx):
        x = input[0]           # [B, T, C]
        x.retain_grad()
        inputs[blk_idx] = x

    for i, blk in enumerate(m.blocks):
        hooks.append(
            blk.attn.register_forward_hook(
                lambda module, inp, out, blk_idx=i: hook_fn(module, inp, out, blk_idx)
            )
        )

    # patch mamba forward exactly like your get_gradients_mamba
    original_forwards = []
    for blk in m.blocks:
        original_forwards.append(blk.attn.forward)

        def new_forward(self, input):
            x_in = input                              # [B, T, C]
            B0, N, C = x_in.shape
            H, Dh = self.head_num, self.head_dim

            x_fwd, _ = self.mamba_fwd(x_in)          # [B, T, C]
            x_bwd, _ = self.mamba_bwd(torch.flip(x_in, dims=[1]))  # [B, T, C]
            x_bwd = torch.flip(x_bwd, dims=[1])
            x_fwd = x_fwd.contiguous()
            x_bwd = x_bwd.contiguous()

            x_fwd_h = x_fwd.reshape(B0, N, H, Dh)
            x_bwd_h = x_bwd.reshape(B0, N, H, Dh)
            x_cat = torch.cat([x_fwd_h, x_bwd_h], dim=-1).contiguous()    # [B, N, H, 2Dh]
            x_cat = x_cat.permute(0, 2, 3, 1).reshape(B0, 2 * C, N).contiguous()
            x_hp = self.head_proj(x_cat)  # [B, C, N]
            x_hp = x_hp.permute(0, 2, 1).contiguous()  # [B, N, C]

            self.attn_out = x_hp  # IMPORTANT: no clone, keep graph
            out = self.post_proj(x_hp)               # [B, T, C]
            return out, self.attn_out

        blk.attn.forward = new_forward.__get__(blk.attn, blk.attn.__class__)

    scores_per_layer = None

    with torch.enable_grad():
        imgs_req = imgs.detach().clone().requires_grad_(True)
        _ = m(imgs_req)

        T = inputs[0].shape[1]
        num_patches = T - 1
        scores_per_layer = [torch.zeros((B, num_patches), dtype=torch.float32) for _ in range(L)]

        for b in range(B):
            for blk_idx in range(L):
                blk_in = inputs[blk_idx]  # [B, T, C]

                m.zero_grad(set_to_none=True)
                if blk_in.grad is not None:
                    blk_in.grad.zero_()

                cls_feat = m.blocks[blk_idx].attn.attn_out[b, 0]  # [C] (sample b, CLS)
                retain = not (b == B - 1 and blk_idx == L - 1)
                cls_feat.sum().backward(retain_graph=retain)

                token_scores = blk_in.grad.detach().abs().sum(dim=-1)[b]  # [T] only sample b
                scores_per_layer[blk_idx][b] = token_scores[1:].cpu()

    # cleanup
    for h in hooks:
        h.remove()
    for blk, f in zip(m.blocks, original_forwards):
        blk.attn.forward = f

    m.eval()
    return [s.contiguous() for s in scores_per_layer]


def get_lstm_cls_scores(model, imgs):
    """
    Interface/return matches get_attn_cls_scores / get_mamba_cls_scores:

    Returns:
        scores_per_layer: list length L
            each tensor shape [B, num_patches] on CPU (excluding CLS)
    """
    model.train()
    m = model.module if hasattr(model, "module") else model

    L = len(m.blocks)
    B = imgs.shape[0]

    inputs = [None] * L
    hooks = []

    def hook_fn(module, input, output, blk_idx):
        x = input[0]              # [B, T, C]
        x.retain_grad()
        inputs[blk_idx] = x

    # hook LSTM input (same as your original code)
    for i, blk in enumerate(m.blocks):
        hooks.append(
            blk.attn.lstm.register_forward_hook(
                lambda module, inp, out, blk_idx=i: hook_fn(module, inp, out, blk_idx)
            )
        )

    # monkey-patch block.attn.forward (same structure as your original, but no clone)
    original_forwards = []
    for blk in m.blocks:
        original_forwards.append(blk.attn.forward)

        def new_forward(self, input):
            # lstm_out: [B, T, 2*hidden_dim_total]  (implementation-specific)
            lstm_out, _ = self.lstm(self.pre_proj(input))
            H = self.hidden_dim // 3
            lstm_out = self.head_proj(lstm_out)

            # IMPORTANT: keep graph, no clone
            self.lstm_out = lstm_out

            return self.post_proj(lstm_out), self.lstm_out

        blk.attn.forward = new_forward.__get__(blk.attn, blk.attn.__class__)

    # allocate after knowing num_patches
    scores_per_layer = None

    with torch.enable_grad():
        imgs_req = imgs.detach().clone().requires_grad_(True)
        _ = m(imgs_req)

        T = inputs[0].shape[1]
        num_patches = T - 1
        scores_per_layer = [torch.zeros((B, num_patches), dtype=torch.float32) for _ in range(L)]

        for b in range(B):
            for blk_idx in range(L):
                blk_in = inputs[blk_idx]  # [B, T, C]

                m.zero_grad(set_to_none=True)
                if blk_in.grad is not None:
                    blk_in.grad.zero_()

                # CLS feature as target (sample b, CLS token 0)
                cls_feat = m.blocks[blk_idx].attn.lstm_out[b, 0]  # [C']

                retain = not (b == B - 1 and blk_idx == L - 1)
                cls_feat.sum().backward(retain_graph=retain)

                token_scores = blk_in.grad.detach().abs().sum(dim=-1)[b]  # [T]
                scores_per_layer[blk_idx][b] = token_scores[1:].cpu()

    # cleanup
    for h in hooks:
        h.remove()
    for blk, f in zip(m.blocks, original_forwards):
        blk.attn.forward = f

    m.eval()
    return [s.contiguous() for s in scores_per_layer]


@torch.no_grad()
def get_attn_cls_scores(model, imgs):
    """
    Returns:
        scores_per_layer: list length L
            each tensor shape [B, num_patches] on CPU
            where num_patches = N-1 (excluding CLS)
    """
    model.eval()
    scores_per_layer = []

    def patched_forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # ---- 只记录 CLS -> patch 的分数：mean over heads ----
        cls2patch = attn[:, :, 0, 1:].mean(dim=1)  # [B, num_patches]
        scores_per_layer.append(cls2patch.detach().cpu())

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    backups = []
    for blk in model.blocks:
        backups.append(blk.attn.forward)
        blk.attn.forward = patched_forward.__get__(blk.attn, blk.attn.__class__)

    _ = model(imgs)

    # restore
    for blk, bk in zip(model.blocks, backups):
        blk.attn.forward = bk

    return scores_per_layer


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUPRC (Average Precision), numpy-only implementation.
    y_true: {0,1} shape [M]
    y_score: float shape [M]
    Returns: AP in [0,1] (nan if no positive)
    """
    y_true = y_true.astype(np.int32)
    if y_true.sum() == 0:
        return float("nan")

    # sort by score descending
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]

    # cumulative true positives
    tp = np.cumsum(y_true_sorted, dtype=np.int64)
    fp = np.cumsum(1 - y_true_sorted, dtype=np.int64)

    precision = tp / (tp + fp + 1e-12)

    # AP = mean precision at each true positive (equivalent to area under PR step curve)
    ap = precision[y_true_sorted == 1].mean()
    return float(ap)


@torch.no_grad()
def compute_sparse_overlap(
    data_loader,
    dense_model,
    avit_model,
    device,
    fixed_img_pairs_path,
    L=12,
    use_amp=True,
):
    fixed_img_pairs = np.load(fixed_img_pairs_path)
    fixed_set = set((int(c), int(i)) for c, i in fixed_img_pairs)
    print(f"Loaded fixed_img_pairs: {len(fixed_set)} samples")

    dense_model.eval()
    avit_model.eval()

    overlap_ratios = [[] for _ in range(L)]
    keep_counts = [[] for _ in range(L)]

    # store as torch tensors on CPU, avoid per-sample numpy() overhead
    auprc_scores = [[] for _ in range(L)]  # list of [num_patches] float tensors
    auprc_labels = [[] for _ in range(L)]  # list of [num_patches] int/bool tensors

    total_patches = None
    matched = 0

    autocast_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else None

    for images, target, indices in data_loader:
        # -------- 0) CPU side filtering (no GPU work before knowing hit) --------
        target_cpu = target.detach().cpu().numpy()
        indices_cpu = np.asarray(indices)

        hit_pos = []
        for b in range(len(indices_cpu)):
            if (int(target_cpu[b]), int(indices_cpu[b])) in fixed_set:
                hit_pos.append(b)

        if len(hit_pos) == 0:
            continue

        matched += len(hit_pos)

        # build sub-batch: only hit samples
        images_hit = images[hit_pos].to(device, non_blocking=True)

        # -------- 1) AViT forward on sub-batch --------
        if autocast_ctx is not None:
            with autocast_ctx():
                _ = avit_model(images_hit)
        else:
            _ = avit_model(images_hit)

        cnt_token = avit_model.counter_token.detach()  # [Bh, N]
        Bh, N = cnt_token.shape
        num_patches = N - 1
        if total_patches is None:
            total_patches = num_patches

        # move depths to CPU once
        depths_cpu = cnt_token[:, 1:].cpu()  # [Bh, num_patches]

        # -------- 2) dense_model scores on sub-batch --------
        # scores_per_layer = get_attn_cls_scores(dense_model, images_hit)  # list[L], each [Bh, num_patches] on CPU
        # scores_per_layer = get_mamba_cls_scores(dense_model, images_hit)  # list[L], each [Bh, num_patches] on CPU
        scores_per_layer = get_lstm_cls_scores(dense_model, images_hit)  # list[L], each [Bh, num_patches] on CPU
        assert len(scores_per_layer) == L, f"Expected L={L}, got {len(scores_per_layer)}"

        # -------- 3) overlap & AUPRC data (only Bh samples) --------
        for l in range(L):
            alive = (depths_cpu >= (l + 1))          # [Bh, num_patches] bool, CPU
            scores_l = scores_per_layer[l]           # [Bh, num_patches] CPU float tensor

            for b in range(Bh):
                kept_mask = alive[b]                 # [num_patches] bool
                k = int(kept_mask.sum().item())
                keep_counts[l].append(k)
                if k == 0:
                    continue

                topk_idx = torch.topk(scores_l[b], k, largest=True).indices  # [k] CPU

                # faster than nonzero+isin: build topk mask then & 
                topk_mask = torch.zeros_like(kept_mask, dtype=torch.bool)
                topk_mask[topk_idx] = True
                inter = (kept_mask & topk_mask).sum().item()
                overlap_ratios[l].append(inter / k)

                # AUPRC: store torch tensors, cat later
                auprc_labels[l].append(kept_mask.to(torch.int8))
                auprc_scores[l].append(scores_l[b].to(torch.float32))

    print(f"Matched fixed samples seen in loader: {matched} / {len(fixed_set)}")

    # -------- aggregate --------
    layer_mean = np.zeros(L, dtype=np.float64)
    layer_var  = np.zeros(L, dtype=np.float64)
    layer_std  = np.zeros(L, dtype=np.float64)
    layer_keep_ratio_mean = np.zeros(L, dtype=np.float64)
    layer_k_mean = np.zeros(L, dtype=np.float64)
    layer_auprc = np.zeros(L, dtype=np.float64)

    for l in range(L):
        # AUPRC
        if len(auprc_labels[l]) == 0:
            layer_auprc[l] = np.nan
        else:
            y_all = torch.cat(auprc_labels[l], dim=0).numpy()
            s_all = torch.cat(auprc_scores[l], dim=0).numpy()
            layer_auprc[l] = average_precision_score(y_all, s_all)

        # overlap stats
        r = np.array(overlap_ratios[l], dtype=np.float64)
        if r.size == 0:
            layer_mean[l] = np.nan
            layer_var[l]  = np.nan
            layer_std[l]  = np.nan
        else:
            layer_mean[l] = r.mean()
            layer_var[l]  = r.var()
            layer_std[l]  = r.std()

        # keep stats
        ks = np.array(keep_counts[l], dtype=np.float64)
        if ks.size == 0 or total_patches is None:
            layer_k_mean[l] = np.nan
            layer_keep_ratio_mean[l] = np.nan
        else:
            layer_k_mean[l] = ks.mean()
            layer_keep_ratio_mean[l] = ks.mean() / float(total_patches)

    stats = {
        "layer_mean_overlap": layer_mean,
        "layer_var_overlap": layer_var,
        "layer_std_overlap": layer_std,
        "layer_mean_k": layer_k_mean,
        "layer_mean_keep_ratio": layer_keep_ratio_mean,
        "overall_mean_overlap": np.nanmean(layer_mean),
        "overall_mean_keep_ratio": np.nanmean(layer_keep_ratio_mean),
        "matched_fixed_seen": matched,
        "fixed_total": len(fixed_set),
        "layer_auprc": layer_auprc,
        "overall_mean_auprc": np.nanmean(layer_auprc),
    }

    print("===== Fixed-set AViT kept vs top-k overlap =====")
    print(f"Fixed matched (seen in loader): {matched}/{len(fixed_set)}")
    print(f"Overall mean overlap (avg over layers): {stats['overall_mean_overlap']:.4f}")
    print(f"Overall mean keep ratio (avg over layers): {stats['overall_mean_keep_ratio']:.4f}")
    print("")
    for l in range(L):
        print(
            f"Layer {l+1:02d} | "
            f"overlap mean={layer_mean[l]:.4f}, var={layer_var[l]:.6f}, std={layer_std[l]:.4f} | "
            f"k_mean={layer_k_mean[l]:.2f}, keep_ratio_mean={layer_keep_ratio_mean[l]:.4f} | "
            f"AUPRC={layer_auprc[l]:.4f} (rand~{layer_keep_ratio_mean[l]:.4f})"
        )

    return stats


def select_fixed_examples(dataset, per_class=1):
    counts = defaultdict(int)
    selected = []

    for idx, (_, cls) in enumerate(dataset.samples):
        if counts[cls] < per_class:
            selected.append((cls, idx))
            counts[cls] += 1

    return selected


def _get_images_from_batch(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, dict):
        for k in ("images", "image", "x", "input"):
            if k in batch:
                return batch[k]
        for v in batch.values():
            if torch.is_tensor(v):
                return v
        raise ValueError("Dict batch has no tensor-like images field.")
    if torch.is_tensor(batch):
        return batch
    raise TypeError(f"Unsupported batch type: {type(batch)}")


@torch.no_grad()
def measure_throughput(
    model,
    dataloader,
    batch_size: int,
    iters,
    warmup,
    device,
):
    it = iter(dataloader)

    def next_images():
        nonlocal it
        while True:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dataloader)
                batch = next(it)
            x = _get_images_from_batch(batch)
            if x.shape[0] >= batch_size:
                return x

    # ---- prefetch one batch to GPU and reuse it ----
    x_big = next_images()
    x0 = x_big[:batch_size].contiguous().to(device, non_blocking=True)

    # warmup (reuse x0)
    for _ in range(warmup):
        _ = model(x0)
    torch.cuda.synchronize()

    # timed (reuse x0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = model(x0)
    torch.cuda.synchronize()
    end = time.perf_counter()

    sec = end - start
    samples_per_sec = (iters * batch_size) / sec
    return samples_per_sec, sec


def make_random_token_dataloader(batch_size: int, N: int, C: int, num_batches: int,
                                 dtype=torch.float16, device: str = "cuda"):
    x = torch.randn(num_batches * batch_size, N, C, dtype=dtype, device=device)
    ds = TensorDataset(x)  # so _get_images_from_batch(batch) returns batch[0]
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)


if __name__ == "__main__":
    parser = config.get_full_parser()
    args = parser.parse_args()
    args = config.fill_default_args(args)

    if args.stat_mode == "sparsity":
        val_base = datasets.ImageFolder(
            "/srv/datasets/imagenet/val/",
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        )
        val_dataset = IndexedDataset(val_base)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        deit_model = create_model(
                model_name=args.base_model_name,
                pretrained=False,
                num_classes=args.nb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                img_size=args.input_size,
            )
        deit_model = architectures.load_weight(deit_model, args.base_weight)
        deit_model.to(args.device)
        avit_model = torch.load(args.stat_weight)
        avit_model.to(args.device)
        compute_sparse_overlap(
            val_loader,
            dense_model=deit_model,
            avit_model=avit_model,
            device=args.device,
            fixed_img_pairs_path=FIXED_IMG_PAIRS_PATH,
        )

    elif args.stat_mode == "throughput":
        # Load model
        print(f"Creating model: {args.stat_model}")
        print(f"Using weight: {args.stat_weight}")
        if "DeiT" in args.stat_model:
            model = create_model(
                model_name=args.stat_model_name,
                pretrained=False,
                num_classes=args.nb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                img_size=args.input_size,
            )
            model = architectures.replace_attention(
                args=args,
                model=model,
                repl_blocks=args.replace,
                target="attn",
            )
            model = architectures.load_weight(model, args.stat_weight)
        else:
            model_args = args if args.avit else None  # For AViT
            model = create_model(
                model_name=args.base_model_name,
                pretrained=False,
                num_classes=args.nb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                img_size=args.input_size,
                args=model_args,
            )
            model = architectures.replace_attention(
                args=args,
                model=model,
                repl_blocks=args.replace,
                target=args.rep_by,
            )
            print(f"Creat from base model {args.base_model_name},\
                  replaced by {args.rep_by}")

        model.to(args.device)

        batch_sizes = [1, 8, 16, 32]
        end2end = False 
        module = True 
        assert not (end2end and module)
        assert end2end or module

        if "384" in args.stat_model:
            N = 577
        else:
            N = 197
        if "Base" in args.stat_model:
            C=768
        else:
            C=192
        N = int(1025 * 0.75)
        if module:
            model = model.blocks[0].attn
            model.eval()
            data_loader = make_random_token_dataloader(
                batch_size=32, N=N, C=C, num_batches=400, dtype=torch.float32, device="cuda"
            )
        if end2end:
            model.eval()
            data_loader, _ = load_dataset(args, "val")
        iterations = 300
        warmup = 50
        for bsz in batch_sizes:
            throughput, total_time = measure_throughput(
                model,
                data_loader,
                bsz,
                iters=iterations,
                warmup=warmup,
                device=args.device,
            )
            print(
                f"Batch size {bsz}: {throughput:8.2f} samples/s in {total_time:.2f} sec"
            )