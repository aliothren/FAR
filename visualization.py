import os
import torch
import models
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from data import load_dataset
from timm.models import create_model

OUTPUT_DIR = "../figs"
MODEL_PATH = {
    "DeiT-Tiny": {
        "name": "deit_tiny_patch16_224",
        "weight": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        },
    "DeiT-Small": {
        "name": "deit_small_patch16_224",
        "weight": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        },
    "DeiT-Base": {
        "name": "deit_base_patch16_224",
        "weight": "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        },
    "LSTM-Tiny": {
        "weight":  "/home/yuxinr/AttnDistill/models/2025-03-07-23-15-10/model_ft_head.pth",
        },
    }
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = {
    "IMNET": "/srv/datasets/imagenet/",
    }


def get_args_parser():
    parser = argparse.ArgumentParser("visualization", add_help=False)
    
    # Environment setups
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--output-dir", default='', help="Output path")
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    
    # Transformer (DeiT) model setups
    parser.add_argument("--base-model", default="DeiT-Tiny", choices=["DeiT-Tiny", "DeiT-Small", "DeiT-Base"],
                        type=str, metavar="MODEL")
    parser.add_argument("--scale", default="Tiny", choices=["Tiny", "Small", "Base"],
                        type=str, metavar="MODEL")
    parser.add_argument("--input-size", default=224, type=int, help="expected images size for model input")
    parser.add_argument("--base-weight", default="", help="path of base model checkpoint")
    parser.add_argument("--attn-weight", default="", help="path of attn part pretrained replace structure")
    
    # data parameters
    parser.add_argument("--dataset", default="IMNET", type=str, 
                        choices=["IMNET", "CIFAR10", "CIFAR100", "INAT18", "INAT19", "FLOWER", "CAR"])
    parser.add_argument("--ft-dataset", default="IMNET", type=str, 
                        choices=["IMNET", "CIFAR10", "CIFAR100", "INAT18", "INAT19", "FLOWER", "CAR"])
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument("--data-path", type=str, help="Path of dataset")
    parser.add_argument("--nb-classes", default=1000, type=int, 
                        help="Number of classes in dataset (default:1000)")
    parser.add_argument("--train-subset", default=1.0, type=float, help="Sampling rate from dataset")
    
    # Running mode
    parser.add_argument("--mode", default="train", choices=["train", "eval", "finetune", "downstream"], 
                        help="Runing mode")
    
    # Training setups
    parser.add_argument("--train-mode", default="parallel", choices=["parallel", "sequential"])
    parser.add_argument("--step", default=12, type=int, help="Step length when sequentially replace blocks and training")
    parser.add_argument("--interm-model", default="", type=str, help="Path of intermediate model in sequential training")
    parser.add_argument("--rep-by", default="lstm", choices=["mixer", "lstm"], 
                        help="Structure used to replace attention")
    parser.add_argument("--skip-train-attn", action='store_true', 
                        help="Use pretrained attn part instead of train from scratch")
    parser.add_argument("--block-ft", action='store_true', 
                        help="Block-level finetune the replaced blocks after training attention")
    parser.add_argument("--reg-in-train", action='store_true', 
                        help="Adding regularization in attn training")
    parser.add_argument("--ds-in-train", action='store_true', 
                        help="Downstream training from deit downstream model")
    parser.set_defaults(block_ft=True)
    parser.add_argument("--train-loss", default="combine", choices=["similarity", "classification", "combine"],
                        type=str, help="Criterion using in training")
    
    # Training parameters
    parser.add_argument('--seed', default=42, type=int, help="Random seed")
    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT",help="Dropout rate (default: 0.)")
    parser.add_argument("--drop-path", type=float, default=0.1, metavar="PCT", help="Drop path rate (default: 0.1)")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=200, type=int, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, metavar='LR', help='Replaced attention learning rate')
    parser.add_argument('--unscale-lr', action='store_true', help="Not scale lr according to batch size")
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, help='Warm-up initial learning rate')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')    
    
    parser.add_argument("--block-ft-mode", default="block", choices=["block", "FC", "FC+head"],
                        type=str, help="Finetune scope in blockwise training")
    parser.add_argument("--block-ft-train-loss", default="classification", choices=["classification", "combine"],
                        type=str, help="Criterion using in blockwise training")
    parser.add_argument("--block-ft-batch-size", default=256, type=int, help="Batch size when block-level finetuning")
    parser.add_argument("--block-ft-epochs", default=100, type=int, help="Training epochs when block-level finetuning")
    parser.add_argument("--block-ft-lr", type=float, default=5e-5, metavar='LR', 
                        help='Learning rate when block-level finetuning')
    parser.add_argument('--block-ft-unscale-lr', action='store_true',
                        help="Not scale lr according to batch size when block-level finetuning")
    parser.add_argument('--block-ft-warmup-epochs', type=int, default=5, 
                        help='Number of warmup epochs when block-level finetuning')
    parser.add_argument('--block-ft-warmup-lr', type=float, default=1e-6,
                        help='Warm-up initial learning rate when block-level finetuning')
    parser.add_argument('--block-ft-sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler when block-level finetuning (default: "cosine")')
    parser.add_argument("--finetune-head", action='store_true', 
                        help="Only use for sequential train/finetune: finetune classification head after sequential train")
    parser.add_argument("--lora", action='store_true', 
                        help="Use lora in block finetune")
    parser.set_defaults(finetune_head=True)
    
    # Evaluation setups
    parser.add_argument("--eval-model", default="", help="Path of model to be evaluated")
    
    # Downstream setups
    parser.add_argument("--ds-mode", default="full", choices=["full", "FC", "FC+head"])
    
    # Finetuning setups
    parser.add_argument("--ft-mode", default="head", choices=["head", "sequential"])
    parser.add_argument("--ft-model", default="", help="Path of model to be finetuned")
    parser.add_argument("--ft-loss", default="classification", 
                        choices=["similarity", "classification", "combine"],
                        type=str, help="Criterion using in global finetune")
    parser.add_argument("--ft-batch-size", default=256, type=int, help="Batch size when global finetuning")
    parser.add_argument("--ft-epochs", default=30, type=int, help="Training epochs when global finetuning")
    parser.add_argument("--ft-lr", type=float, default=5e-4, metavar='LR', 
                        help='Learning rate when global finetuning')
    parser.add_argument('--ft-unscale-lr', action='store_true',
                        help="Not scale lr according to batch size when global finetuning")
    parser.add_argument('--ft-warmup-epochs', type=int, default=5, 
                        help='Number of warmup epochs when global finetuning')
    parser.add_argument('--ft-warmup-lr', type=float, default=1e-5,
                        help='Warm-up initial learning rate when global finetuning')
    parser.add_argument('--ft-sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler when global finetuning (default: "cosine")')
    
    # data augment parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument("--repeated-aug", action="store_true", help="used in distributed training")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
    parser.set_defaults(repeated_aug=True)
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # pruning setups
    parser.add_argument('--arch', '-a', metavar='ARCH', default='LSTM',
                        choices=["LSTM", "Mixer"], help='pretrained model architecture')    
    parser.add_argument('--reg', type=int, default=3, metavar='R',
                        help='regularization type: 0:None 1:L1 2:Hoyer 3:HS')
    parser.add_argument('--decay', type=float, default=1e-4, metavar='D',
                        help='weight decay for regularizer (default: 0.001)')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')  
    parser.add_argument('--sensitivity', type=float, default=1e-4, help="threshold used for pruning")
    
    return parser
    

def plot_heatmap(
    data: torch.Tensor, 
    title: str, 
    save_path=None,
    save_cls = True,
    save_patch = True
) -> None:
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    seq_length = data.shape[0]

    fig, ax = plt.subplots()
    ax.imshow(data, cmap='viridis', vmin=0, vmax=5)
    ax.set_xticks([0, seq_length - 1])
    ax.set_xticklabels(['0', f'{seq_length - 1}'])
    ax.set_yticks([0, seq_length - 1])
    ax.set_yticklabels(['0', f'{seq_length - 1}'])

    plt.title(title)
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')

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
        ax.set_ylabel("Attention Weight")
        ax.set_xlim(0, len(cls_data) - 1)
        ax.set_ylim(0, 1)
        ax.grid(True)

        plt.savefig(cls_path, dpi=300, bbox_inches='tight')
        print(f"CLS Attention saved to {cls_path}")
        plt.close(fig)
    
    if save_patch and save_path:
        patch_data = data[0, 1:].reshape(14, 14)  # drop CLS token
        patch_path = save_path.with_name(save_path.stem + "_cls_patch" + save_path.suffix)

        fig, ax = plt.subplots()
        im = ax.imshow(patch_data, cmap='viridis', vmin=0, vmax=0.5)
        ax.set_title(title + " (CLS → Patches)")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)

        plt.savefig(patch_path, dpi=300, bbox_inches='tight')
        print(f"Patch Attention saved to {patch_path}")
        plt.close(fig)
        

def plot_attention_heatmap(attentions, head_ids, layer_ids, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for layer_id in layer_ids:
        attention = attentions[layer_id]
        for head_id in head_ids:
            data = attention[head_id].numpy()
            save_path = save_dir / f"layer{layer_id}_head{head_id}.png"
            plot_heatmap(data, f'Attention Map: Layer {layer_id} Head {head_id}', save_path)
        avg_data = torch.mean(attention, dim=0).numpy()
        save_path = save_dir / f"layer{layer_id}_avg.png"
        plot_heatmap(avg_data, f'Average Attention Map: Layer {layer_id}', save_path)


def plot_gradiant_heatmap(gradiants, layer_ids, save_dir, mode):
    os.makedirs(save_dir, exist_ok=True)
    for layer_id in layer_ids:
        gradiant = gradiants[layer_id]
        data = gradiant.numpy()
        save_path = save_dir / f"layer{layer_id}_{mode}.png"
        plot_heatmap(data, f'Average Attention Map: Layer {layer_id}', save_path)


def get_block_gradiants(model, imgs):
    model.train() 
    hooks = []
    gradiants = []
    inputs = [None for _ in range(len(model.blocks))]
    outputs = [None for _ in range(len(model.blocks))]
    
    for i, blk in enumerate(model.blocks):
        def hook_fn(module, input, output, blk_idx=i):
            input = input[0]
            input.retain_grad()
            output = output[0]
            inputs[blk_idx] = input
            outputs[blk_idx] = output
        hook = blk.register_forward_hook(hook_fn)
        hooks.append(hook)
           
    for b in range(imgs.shape[0]): # batchsize
        img = imgs[b].unsqueeze(0).detach().clone().requires_grad_(True)
        inputs = [None for _ in range(len(model.blocks))]
        outputs = [None for _ in range(len(model.blocks))]
        with torch.enable_grad():
            _ = model(img)
        gradiant = []
        for blk_idx in range(len(model.blocks)):
            blk_input = inputs[blk_idx] 
            blk_output = outputs[blk_idx] 
            token_num = blk_output.shape[0]

            layer_grad = torch.zeros((token_num, token_num))
            for token_idx in range(token_num):
                model.zero_grad()
                if blk_input.grad is not None:
                    blk_input.grad.zero_()
                token = blk_output[token_idx]
                token.sum().backward(retain_graph=True)
                token_grad = blk_input.grad.detach().abs().sum(dim=-1).squeeze(0) 
                layer_grad[token_idx] = token_grad.clone().cpu() 
            gradiant.append(layer_grad) 
         
        gradiants.append(torch.stack(gradiant))
    gradiants = torch.stack(gradiants, dim=0)
    avg_gradiant = gradiants.mean(dim=0)
    for h in hooks:
        h.remove()

    return avg_gradiant   
        

def get_gradiants(model, imgs, mode="avg"):
    model.eval()
    
    hooks = []
    original_forwards = []
    inputs = [None for _ in range(len(model.blocks))]
    outputs = [None for _ in range(len(model.blocks))]
        
    for i, blk in enumerate(model.blocks):
        def hook_fn(module, input, output, blk_idx=i):
            input = input[0]
            input.retain_grad()
            inputs[blk_idx] = input
        hooks.append(blk.attn.lstm.register_forward_hook(hook_fn))
        original_forward = blk.attn.forward
        original_forwards.append(original_forward)
        
        def new_forward(self, input, blk_idx=i):
            H = self.lstm.hidden_size
            if mode == "avg":
                out = original_forwards[blk_idx](input)
                outputs[blk_idx] = out
            elif mode == "backward":
                out, _ = self.lstm(input)
                outputs[blk_idx] = torch.flip(out[:, :, H:], dims=[1])
                out = self.proj(out) 
            elif mode == "forward":
                out, _ = self.lstm(input)
                outputs[blk_idx] = out[:, :, :H]
                out = self.proj(out)
                
            return out
        
        blk.attn.forward = new_forward.__get__(blk.attn, blk.attn.__class__)
        
    gradiants = []
    for b in range(imgs.shape[0]): # batchsize
        model.train() 
        img = imgs[b].unsqueeze(0).detach().clone().requires_grad_(True)
        inputs = [None for _ in range(len(model.blocks))]
        outputs = [None for _ in range(len(model.blocks))]
        with torch.enable_grad():
            _ = model(img)
        
        gradiant = []
        for blk_idx in range(len(model.blocks)):
            blk_input = inputs[blk_idx] # [1, T, D]
            blk_output = outputs[blk_idx] # [1, T, D]
            token_num = blk_output.shape[1]
            
            layer_grad = torch.zeros((token_num, token_num))
            for token_idx in range(token_num):
                model.zero_grad()
                token = blk_output[0, token_idx]
                token.sum().backward(retain_graph=True)
                token_grad = blk_input.grad.detach().abs().sum(dim=-1).squeeze(0) 
                layer_grad[token_idx] = token_grad.cpu() 
            layer_grad = torch.flip(layer_grad, dims=[0, 1])
            gradiant.append(layer_grad) 
            
        model.eval()
        gradiants.append(torch.stack(gradiant)) # [L, T, T]
    
    gradiants = torch.stack(gradiants, dim=0)
    avg_gradiant = gradiants.mean(dim=0)
    
    for h in hooks:
        h.remove()
    for i, blk in enumerate(model.blocks):
        blk.attn.forward = original_forwards[i]

    return avg_gradiant   
            

def get_attentions(model, imgs):
    hooks = []
    attn_scores = []
    model.eval()
    # monkey patch Attention.forward
    def hook_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_scores.append(attn.detach().cpu())  # <<< 记录 softmax 后的 attention
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    # 替换所有 blocks 的 attn.forward
    backup = []
    for blk in model.blocks:
        backup.append(blk.attn.forward)
        blk.attn.forward = hook_forward.__get__(blk.attn, blk.attn.__class__)  # bind 方法

    with torch.no_grad():
        _ = model(imgs)

    for h in hooks:
        h.remove()
        
    for i in range(len(attn_scores)):
        attn = attn_scores[i]  # shape: [B, H, N, N]
        if attn.shape[0] > 1:  # 多张图
            attn_scores[i] = attn.mean(dim=0)  # → [H, N, N]
        else:  # 单张图
            attn_scores[i] = attn[0]  # 去掉 batch 维 → [H, N, N]
    return attn_scores                 


if __name__ == "__main__":
    parser = argparse.ArgumentParser("visualization", parents=[get_args_parser()])
    args = parser.parse_args()
    
    layers = list(range(12))
    heads = [0, 1, 2]
    batch = 32
    
    # Load data
    if args.data_path is None:
        args.data_path = DATA_PATH[args.dataset]
    data_loader_val, _ = load_dataset(args, "val")
    imgs, targets = next(iter(data_loader_val)) 
    imgs = imgs[0:batch]
    imgs = imgs.to(args.device)
    
    # Load model
    model_name = "LSTM-Tiny"
    print(f"Creating model: {model_name}")
    model_path = MODEL_PATH[model_name]["weight"]
    print(f"Using weight: {model_path}")
    if "DeiT" in model_name:
        model = create_model(
            model_name=MODEL_PATH[model_name]["name"], pretrained=False, num_classes=args.nb_classes, drop_rate=args.drop,
            drop_path_rate=args.drop_path, drop_block_rate=None, img_size=args.input_size
            )
        model = models.load_weight(model, model_path)
    else:
        model = torch.load(model_path)
    model.to(args.device)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # # Get attention scores
    # save_path = Path(OUTPUT_DIR) / f"{model_name}"
    # if "DeiT" in model_name:
    #     attentions =  get_attentions(model, imgs) 
    #     # Save heatmap
    #     plot_attention_heatmap(attentions, heads, layers, save_path)
    # else:
    #     mode = "backward"
    #     gradiants = get_gradiants(model, imgs, mode) 
    #     plot_gradiant_heatmap(gradiants, layers, save_path, mode)
        
    # Get blockwise gradiants
    block_gradiants = get_block_gradiants(model, imgs)
    save_path = Path(OUTPUT_DIR) / "batch32" / f"{model_name}_block"
    plot_gradiant_heatmap(block_gradiants, layers, save_path, "grad")
    