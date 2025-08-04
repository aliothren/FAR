import json
import copy
import torch
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from timm.data import Mixup
from timm.utils import ModelEma
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

from modeling import utils
from modeling import config
from modeling import architectures
from modeling.data import load_dataset
from modeling.train import train_model, evaluate_model, compute_lstm_reg_multihead


def prune(args, model, data_loader_val):
    # Initial pruning
    print("--- Pruning ---")
    actual_model = model.module if hasattr(model, "module") else model
    masks = {}
    if args.rep_by == "multi-lstm":
        for blk_idx, block in enumerate(actual_model.blocks):
            block_masks = compute_lstm_reg_multihead(block, args, mode="mask")
            block_masks = {f"blocks.{blk_idx}.{k}": v for k, v in block_masks.items()}
            masks.update(block_masks)

    else:
        for name, p in model.named_parameters():
            if "weight" in name:
                tensor = p.data.cpu().numpy()
                new_mask = np.where(abs(tensor) < args.sensitivity, 0, tensor)
                mask = np.where(abs(tensor) < args.sensitivity, 0.0, 1.0)
                masks[name] = torch.from_numpy(mask).float().to(args.device)
                p.data = torch.from_numpy(new_mask).to(args.device)

    utils.print_nonzeros(model)
    print("Pruned model evaluation...")
    evaluate_model(data_loader_val, args.device, model)

    return model, masks


def analyze_pruned_heads(model, head_num=3):
    actual_model = model.module if hasattr(model, "module") else model
    for blk_idx, block in enumerate(actual_model.blocks):
        lstm = block.attn.lstm
        # input_dim = block.attn.input_dim
        hidden_dim = block.attn.hidden_dim
        hid_per_head = hidden_dim // head_num
        # in_per_head = input_dim // head_num

        num_directions = 2 if lstm.bidirectional else 1

        for direction in range(num_directions):
            suffix = "_l0" + ("_reverse" if direction == 1 else "")
            weight_name = f"weight_ih{suffix}"
            weight = getattr(lstm, weight_name, None)  # shape: [4*H, In]

            if weight is None:
                continue

            weight = weight.data.cpu().numpy()  # shape: [4*H, In]

            print(
                f"\nBlock {blk_idx} | Direction: {'reverse' if direction == 1 else 'forward'}"
            )
            for h in range(head_num):
                h_start = h * hid_per_head
                h_end = (h + 1) * hid_per_head
                # in_start = h * in_per_head
                # in_end = (h + 1) * in_per_head
                gate_tensors = []
                for g in range(4):  # i, f, g, o
                    start_idx = g * hidden_dim + h_start
                    end_idx = g * hidden_dim + h_end
                    gate_block = weight[
                        start_idx:end_idx, :
                    ]  # shape [hid_per_head, in_per_head]
                    gate_tensors.append(gate_block)
                head_tensor = np.concatenate(gate_tensors, axis=1)

                # check channels pruned by all heads
                is_pruned = np.all(head_tensor == 0, axis=1)
                pruned_count = np.sum(is_pruned)
                print(
                    f"Head {h:2d} | Pruned hidden channels: {pruned_count:2d} / {hid_per_head}"
                )


def main(args):

    print(
        f"Running in Pruning mode,\
          pruning on FAR model pretrained from {args.far_model},\
          replaced by {args.rep_by}"
    )
    print(f"Using device: {args.device}")

    if args.fixed_seed:
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Using seed: {seed}")

    cudnn.benchmark = True

    data_loader_train = load_dataset(args, "train")
    data_loader_val, _ = load_dataset(args, "val")
    mixup_fn = None
    args.mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if args.mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # Load model
    print(f"Loading model: {args.far_weight}")
    model = torch.load(args.far_weight)
    model.to(args.device)
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(),
        # DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else '',
            resume='')
    architectures.set_requires_grad(model, "prune")  # whole model trainable

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable params: {n_parameters}")

    # regularized training
    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    reg_model, reg_model_dict = train_model(
        args=args,
        stage="reg",
        loss_mode="classification",
        model=model,
        teacher_model=None,
        model_ema=model_ema,
        train_data=data_loader_train,
        test_data=data_loader_val,
        mixup_fn=mixup_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        n_parameters=n_parameters,
    )

    if utils.get_rank() == 0:
        save_path = args.output_dir / "model_reg.pth"
        if hasattr(reg_model, "module"):
            torch.save(reg_model.module, save_path)
        else:
            torch.save(reg_model, save_path)

    # pruning
    reg_model_without_ddp = reg_model
    if hasattr(reg_model_without_ddp, "module"):
        reg_model_without_ddp = reg_model_without_ddp.module
    pruned_model, pruning_mask = prune(args, reg_model_without_ddp, data_loader_val)
    try:
        model_cpu = copy.deepcopy(pruned_model).cpu()
        analyze_pruned_heads(model_cpu, model_cpu.blocks[0].attn.head_num)
    except Exception as e:
        print(f"Unable to print pruned heads: {e}")

    # finetuning pruned model
    pruned_model_without_ddp = pruned_model
    pruned_model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(),
        # DP wrapper, and AMP but before SyncBN and DDP wrapper
        pruned_model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else '',
            resume='')
    if hasattr(pruned_model_without_ddp, "module"):
        pruned_model_without_ddp = pruned_model_without_ddp.module
    if args.distributed:
        pruned_model = torch.nn.parallel.DistributedDataParallel(
            pruned_model, device_ids=[args.gpu]
        )

    print("---- Post-pruning finetuning ----")
    brgs = copy.deepcopy(args)
    brgs.lr = args.ft_lr
    brgs.epochs = args.ft_epochs
    brgs.batch_size = args.ft_batch_size

    optimizer = create_optimizer(brgs, pruned_model_without_ddp)
    lr_scheduler, _ = create_scheduler(brgs, optimizer)

    fted_model, ft_model_dict = train_model(
        args=brgs,
        stage="reg_ft",
        loss_mode="classification",
        model=pruned_model,
        teacher_model=None,
        model_ema=pruned_model_ema,
        train_data=data_loader_train,
        test_data=data_loader_val,
        mixup_fn=mixup_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        n_parameters=n_parameters,
        mask=pruning_mask,
    )

    if utils.get_rank() == 0:
        save_path = args.output_dir / "model_ft.pth"
        if hasattr(fted_model, "module"):
            torch.save(fted_model.module, save_path)
        else:
            torch.save(fted_model, save_path)


if __name__ == "__main__":
    parser = config.get_full_parser()
    args = parser.parse_args()
    args.mode = "prune"
    args = config.fill_default_args(args)

    if utils.get_rank() == 0:
        print(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=4))
    utils.init_distributed_mode(args)

    main(args)
