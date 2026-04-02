import os
import cv2
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

from modeling import config
from modeling import models_act
from modeling import architectures
from modeling.data import load_dataset

FIXED_IMG_PAIRS_PATH = "/home/yuxinr/far/FAR/figs/fixed_pairs.npy"


class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.samples = base_dataset.samples

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        img, target = self.base[index]
        return img, target, index
    

def plot_heatmap(
    data: torch.Tensor, title: str, save_path=None, save_cls=True, save_patch=True
) -> None:
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    seq_length = data.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=3.0)
    # im = ax.imshow(np.log1p(data), cmap='viridis', vmin=0, vmax=2.0)
    ax.set_xticks([0, seq_length - 1])
    ax.set_xticklabels(["0", f"{seq_length - 1}"])
    ax.set_yticks([0, seq_length - 1])
    ax.set_yticklabels(["0", f"{seq_length - 1}"])
    fig.colorbar(im, ax=ax, shrink=0.9)

    plt.title(title)
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to {save_path}")
    plt.close(fig)

    if save_cls and save_path:
        cls_data = data[0]
        cls_path = save_path.with_name(save_path.stem + "_cls" + save_path.suffix)

        fig, ax = plt.subplots()
        ax.plot(cls_data)
        ax.set_title(title + " (CLS Row)")
        ax.set_xlabel("Key Token Index")
        ax.set_ylabel("Gradient Score")
        ax.set_xlim(0, len(cls_data) - 1)
        ax.set_ylim(0, 1)
        ax.grid(True)

        plt.savefig(cls_path, dpi=300, bbox_inches="tight")
        print(f"CLS Attention saved to {cls_path}")
        plt.close(fig)

    if save_patch and save_path:
        patch_data = data[0, 1:].reshape(14, 14)  # drop CLS token
        patch_path = save_path.with_name(
            save_path.stem + "_cls_patch" + save_path.suffix
        )

        fig, ax = plt.subplots()
        im = ax.imshow(patch_data, cmap="viridis", vmin=0, vmax=2.0)
        # im = ax.imshow(patch_data, cmap='viridis')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.9)

        plt.savefig(patch_path, dpi=300, bbox_inches="tight")
        print(f"Patch Attention saved to {patch_path}")
        plt.close(fig)


def plot_attention_heatmap(attentions, head_ids, layer_ids, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for layer_id in layer_ids:
        attention = attentions[layer_id]
        for head_id in head_ids:
            data = attention[head_id].numpy()
            save_path = save_dir / f"layer{layer_id}_head{head_id}.png"
            plot_heatmap(data, f"Ours: Layer {layer_id} Head {head_id}", save_path)
        avg_data = torch.mean(attention, dim=0).numpy()
        save_path = save_dir / f"layer{layer_id}_avg.png"
        plot_heatmap(avg_data, f"Gradient Score Map: Layer {layer_id}", save_path)


def plot_gradiant_heatmap(gradiants, layer_ids, save_dir, mode):
    os.makedirs(save_dir, exist_ok=True)
    for layer_id in layer_ids:
        gradiant = gradiants[layer_id]
        data = gradiant.numpy()
        save_path = save_dir / f"layer{layer_id}_{mode}.png"
        plot_heatmap(data, f"Average Attention Map: Layer {layer_id}", save_path)


def get_gradients_mamba(model, imgs, head_num=3, mode="avg"):
    """
    return grad_maps:  shape = [head_num, num_blocks, T, T]
    """
    model.train()

    inputs = [None] * len(model.blocks)
    outputs = [[None] * head_num for _ in model.blocks]
    hooks, original_forwards = [], []

    print("Registering hooks and fwds")
    for idx, blk in enumerate(model.blocks):

        def hook_fn(module, input, output, blk_idx=idx):
            input = input[0]
            input.retain_grad()
            inputs[blk_idx] = input

        hooks.append(blk.attn.register_forward_hook(hook_fn))

        # ---- monkey-patch block.attn.forward ----
        original_forward = blk.attn.forward
        original_forwards.append(original_forward)

        def new_forward(self, input, blk_idx=idx):
            x_in = input                              # [B, T, C]
            B, N, C = x_in.shape
            H, Dh = self.head_num, self.head_dim

            x_fwd, _ = self.mamba_fwd(x_in)          # [B, T, C]
            x_bwd, _ = self.mamba_bwd(torch.flip(x_in, dims=[1]))  # [B, T, C]
            x_bwd = torch.flip(x_bwd, dims=[1])  
            x_fwd = x_fwd.contiguous()
            x_bwd = x_bwd.contiguous()
            # [B, N, H, Dh]
            x_fwd_h = x_fwd.reshape(B, N, H, Dh)
            x_bwd_h = x_bwd.reshape(B, N, H, Dh)
            x_cat = torch.cat([x_fwd_h, x_bwd_h], dim=-1).contiguous()    # [B, N, H, 2Dh]
            x_cat = x_cat.permute(0, 2, 3, 1).reshape(B, 2 * C, N).contiguous()
            x_hp = self.head_proj(x_cat)  # [B, C, N]
            x_hp = x_hp.permute(0, 2, 1).contiguous()  # [B, N, C]

            self.attn_out = x_hp.clone()
            for head_idx in range(head_num):
                s = head_idx * Dh
                e = (head_idx + 1) * Dh
                head_feat = x_hp[:, :, s:e]          # [B, T, Dh]
                outputs[blk_idx][head_idx] = head_feat

            out = self.post_proj(x_hp)               # [B, T, C]
            return out, self.attn_out

        blk.attn.forward = new_forward.__get__(blk.attn, blk.attn.__class__)

    print("Calculating gradiants")
    gradiants = []
    for b in range(imgs.shape[0]):  # batchsize
        img = imgs[b].unsqueeze(0).detach().clone().requires_grad_(True)
        inputs = [None] * len(model.blocks)
        outputs = [[None] * head_num for _ in model.blocks]
        with torch.enable_grad():
            _ = model(img)

        gradiant = [[] for _ in range(head_num)]
        for blk_idx in range(len(model.blocks)):
            blk_input = inputs[blk_idx]
            blk_output = outputs[blk_idx]

            for head_idx in range(head_num):
                print(
                    f"Calculating grad of img {b} for block {blk_idx}, head {head_idx}"
                )
                head_output = blk_output[head_idx]  # [1,T,*]
                token_num = head_output.shape[1]
                layer_grad = torch.zeros((token_num, token_num))

                for token_idx in range(token_num):
                    model.zero_grad()
                    if blk_input.grad is not None:
                        blk_input.grad.zero_()
                    token = head_output[0, token_idx]
                    token.sum().backward(retain_graph=True)
                    token_grad = blk_input.grad.detach().abs().sum(dim=-1).squeeze(0)
                    layer_grad[token_idx] = token_grad.cpu()

                gradiant[head_idx].append(layer_grad)

        gradiants.append(torch.stack([torch.stack(h) for h in gradiant]))
    gradiants = torch.stack(gradiants, dim=0)
    avg_gradiant = gradiants.mean(dim=0).permute(1, 0, 2, 3).contiguous()

    for h in hooks:
        h.remove()
    for i, blk in enumerate(model.blocks):
        blk.attn.forward = original_forwards[i]

    model.eval()
    return avg_gradiant


def get_gradients_multihead(model, imgs, head_num=3, mode="avg"):
    """
    return grad_maps:  shape = [head_num, num_blocks, T, T]
    """
    model.train()

    inputs = [None] * len(model.blocks)
    outputs = [[None] * head_num for _ in model.blocks]
    hooks, original_forwards = [], []

    print("Registering hooks and fwds")
    for idx, blk in enumerate(model.blocks):

        def hook_fn(module, input, output, blk_idx=idx):
            input = input[0]
            input.retain_grad()
            inputs[blk_idx] = input

        hooks.append(blk.attn.lstm.register_forward_hook(hook_fn))

        # ---- monkey-patch block.attn.forward ----
        original_forward = blk.attn.forward
        original_forwards.append(original_forward)

        def new_forward(self, input, blk_idx=idx):
            # lstm_out: [B, T, 2*hidden_dim_total]  (=2*head_num*H)
            lstm_out, _ = self.lstm(self.pre_proj(input))
            H = self.hidden_dim // 3
            lstm_out = self.head_proj(lstm_out)
            self.lstm_out = lstm_out.clone()
            # H = self.hidden_dim // self.head_num

            for head_idx in range(head_num):
                start = head_idx * H
                end = (head_idx + 1) * H
                head_feat = lstm_out[:, :, start:end]          # [B, T, Dh]
                outputs[blk_idx][head_idx] = head_feat

            return self.post_proj(lstm_out), self.lstm_out

        blk.attn.forward = new_forward.__get__(blk.attn, blk.attn.__class__)

    print("Calculating gradiants")
    gradiants = []
    for b in range(imgs.shape[0]):  # batchsize
        img = imgs[b].unsqueeze(0).detach().clone().requires_grad_(True)
        inputs = [None] * len(model.blocks)
        outputs = [[None] * head_num for _ in model.blocks]
        with torch.enable_grad():
            _ = model(img)

        gradiant = [[] for _ in range(head_num)]
        for blk_idx in range(len(model.blocks)):
            blk_input = inputs[blk_idx]
            blk_output = outputs[blk_idx]

            for head_idx in range(head_num):
                print(
                    f"Calculating grad of img {b} for block {blk_idx}, head {head_idx}"
                )
                head_output = blk_output[head_idx]  # [1,T,*]
                token_num = head_output.shape[1]
                layer_grad = torch.zeros((token_num, token_num))

                for token_idx in range(token_num):
                    model.zero_grad()
                    if blk_input.grad is not None:
                        blk_input.grad.zero_()
                    token = head_output[0, token_idx]
                    token.sum().backward(retain_graph=True)
                    token_grad = blk_input.grad.detach().abs().sum(dim=-1).squeeze(0)
                    layer_grad[token_idx] = token_grad.cpu()

                gradiant[head_idx].append(layer_grad)

        gradiants.append(torch.stack([torch.stack(h) for h in gradiant]))
    gradiants = torch.stack(gradiants, dim=0)
    avg_gradiant = gradiants.mean(dim=0).permute(1, 0, 2, 3).contiguous()

    for h in hooks:
        h.remove()
    for i, blk in enumerate(model.blocks):
        blk.attn.forward = original_forwards[i]

    model.eval()
    return avg_gradiant


def get_attentions(model, imgs):
    hooks = []
    attn_scores = []
    model.eval()

    # monkey patch Attention.forward
    def hook_forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_scores.append(attn.detach().cpu())
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    backup = []
    for blk in model.blocks:
        backup.append(blk.attn.forward)
        blk.attn.forward = hook_forward.__get__(blk.attn, blk.attn.__class__)

    with torch.no_grad():
        _ = model(imgs)

    for h in hooks:
        h.remove()

    for i in range(len(attn_scores)):
        attn = attn_scores[i]  # shape: [B, H, N, N]
        if attn.shape[0] > 1:
            attn_scores[i] = attn.mean(dim=0)  # → [H, N, N]
        else:
            attn_scores[i] = attn[0]  # → [H, N, N]
    return attn_scores


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    # snippet for merging and visualization
    h_min = max(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def merge_image(im1, im2):
    # snippet for merging and visualization
    h_margin = 54
    v_margin = 80
    im2 = im2[h_margin+5:480-h_margin, v_margin:640-v_margin]
    return hconcat_resize_min([im1, im2])


def select_fixed_examples(dataset, per_class=1):
    counts = defaultdict(int)
    selected = []

    for idx, (_, cls) in enumerate(dataset.samples):
        if counts[cls] < per_class:
            selected.append((cls, idx))
            counts[cls] += 1

    return selected


@torch.no_grad()
def visualize_avit(data_loader, model, device, file_path, fixed_img_pairs_path=None):
    if fixed_img_pairs_path is None:
        fixed_img_pairs = select_fixed_examples(data_loader.dataset, per_class=1)
        np.save(FIXED_IMG_PAIRS_PATH, np.array(fixed_img_pairs, dtype=np.int64))
        print("Saved", len(fixed_img_pairs), "pairs.")
    else:
        fixed_img_pairs = np.load(fixed_img_pairs_path)
        print("Loaded", len(fixed_img_pairs), "pairs.")

    # this snipet visualize the token depth distribution of an avit model
    # more particular, it saves the image with the largset token depth std. per imagenet class
    # in validation set.

    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluation mode
    model.eval()
    model = model.module if hasattr(model, "module") else model

    # amid imagenet class separation for best visualization, assert batch size is 10
    # such that no validation images overlap in classes
    # assert args.batch_size==50
    fixed_set = set((int(c), int(i)) for c, i in fixed_img_pairs)

    # output reserving rate per layer
    L = 12
    layer_alive_sum = np.zeros(L, dtype=np.float64)
    layer_token_count = 0  #

    for images, target, indices in data_loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            _ = criterion(output, target)

        cnt_token = model.counter_token.data.cpu().numpy()
        B, N = cnt_token.shape
        num_patches = N - 1
        H = W = int(num_patches ** 0.5)
        depths = cnt_token[:, 1:]  # [B, num_patches]
        for l_idx in range(L):
            alive = (depths >= (l_idx + 1))  # bool [B, num_patches]
            layer_alive_sum[l_idx] += alive.sum()

        layer_token_count += B * num_patches

        images_cpu = images.detach().cpu()
        target_cpu = target.detach().cpu().numpy()
        indices_cpu = np.array(indices)

        for b in range(B):
            cls = int(target_cpu[b])
            idx = int(indices_cpu[b])
            if (cls, idx) not in fixed_set:
                continue  # 不是我们选定的样本，跳过

            tokens = cnt_token[b, 1:]  # 去掉CLS
            array = tokens.reshape(H, W)

            plt.figure()
            plt.imshow(
                array, 
                cmap='hot', 
                interpolation='nearest',
                vmin=2,
                vmax=12,          
            )
            plt.axis('off')
            cb = plt.colorbar(shrink=0.8)

            depth_path = os.path.join(file_path, f"class{cls}_idx{idx}_depth.jpg")
            img_path   = os.path.join(file_path, f"class{cls}_idx{idx}_ref.jpg")
            comb_path  = os.path.join(file_path, f"class{cls}_idx{idx}_combined.jpg")

            plt.savefig(depth_path, bbox_inches='tight', pad_inches=0)
            cb.remove()
            plt.close()

            vutils.save_image(
                images_cpu[b],
                img_path,
                normalize=True,
                scale_each=True,
            )

            im1 = cv2.imread(img_path)
            im2 = cv2.imread(depth_path)
            if im1 is not None and im2 is not None:
                h = max(im1.shape[0], im2.shape[0])

                def pad(im):
                    pad_h = h - im.shape[0]
                    if pad_h <= 0:
                        return im
                    return cv2.copyMakeBorder(
                        im, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                    )

                im1p, im2p = pad(im1), pad(im2)
                comb = np.concatenate([im1p, im2p], axis=1)
                cv2.imwrite(comb_path, comb)

    print('Visualization done.')
    layer_keep_ratio = layer_alive_sum / layer_token_count  # [L]

    print("Per-layer average keep ratio (excluding CLS):")
    print(layer_keep_ratio)
    for l_idx in range(L):
        print(f"Layer {l_idx+1}: {layer_keep_ratio[l_idx]:.4f}")

    return


if __name__ == "__main__":
    # Modify visualization scope here
    layers = list(range(12))
    heads = [0, 1, 2]
    batch = 1

    parser = config.get_full_parser()
    args = parser.parse_args()
    args = config.fill_default_args(args)

    # Load model
    print(f"Creating model: {args.vis_model}")
    print(f"Using weight: {args.vis_weight}")
    if "DeiT" in args.vis_model:
        model = create_model(
            model_name=args.vis_model_name,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size,
        )
        model = architectures.load_weight(model, args.vis_weight)
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
        model = architectures.load_weight(model, args.vis_weight)
        # model.use_external_mask = False
        # torch.save(model, "/home/yuxinr/far/FAR/figs/temp_model.pth")
        # exit(0)
        # model = torch.load(args.vis_weight)
    model.to(args.device)

    output_dir = Path(args.base_dir) / "figs"
    save_path = output_dir / f"{args.vis_model}-{args.vis_mode}"
    os.makedirs(save_path, exist_ok=True)

    if args.vis_mode == "token":
        # Load data
        data_loader_val, _ = load_dataset(args, "val")
        imgs, targets = next(iter(data_loader_val))
        imgs = imgs[0:batch]
        imgs = imgs.to(args.device)
        # Get attention scores
        if "DeiT" in args.vis_model:
            save_path = save_path / "uni"
            attentions = get_attentions(model, imgs)
            # Save heatmap
            plot_attention_heatmap(attentions, heads, layers, save_path)
        elif "Multihead" in args.vis_model:
            mode = "avg"
            save_path = save_path / "uni"
            gradiants = get_gradients_multihead(model, imgs, mode=mode)
            plot_attention_heatmap(gradiants, heads, layers, save_path)
        elif "Mamba" in args.vis_model:
            mode = "avg"
            save_path = save_path / "uni"
            gradiants = get_gradients_mamba(model, imgs, mode=mode)
            plot_attention_heatmap(gradiants, heads, layers, save_path)

    elif args.vis_mode == "avit":
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
        visualize_avit(val_loader, model, args.device, save_path, FIXED_IMG_PAIRS_PATH)
