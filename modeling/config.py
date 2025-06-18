import torch
import fcntl
import argparse
import datetime
from pathlib import Path

###### ----------- Global PATHs and configs ----------- ######
# Path to datasets
DATA_PATH = {
    "IMNET": "/contrib/datasets/ILSVRC2012/",
    "CIFAR100": "/home/u17/yuxinr/datasets/CIFAR100",
    "CIFAR10": "/home/u17/yuxinr/datasets/CIFAR10",
    "FLOWER": "/home/u17/yuxinr/datasets/Flowers102",
    "CAR": "/home/u17/yuxinr/datasets/StanfordCars",
    }

# Path to pretrained checkpoints of ATTENTION architecture models
BASE_MODEL_PATH = {
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
    }

# Path to downstreamed checkpoints of ATTENTION architecture models
BASE_MODEL_DS_PATH = {
    "DeiT-Tiny": {
        "CIFAR10": "/benchmodels/deit/tiny_cifar10.pth",
        "CIFAR100": "/benchmodels/deit/tiny_cifar100.pth",
        "FLOWER": "/benchmodels/deit/tiny_flower.pth",
        "CAR": "/benchmodels/deit/tiny_car.pth",
        },
    "DeiT-Small": {
        },
    "DeiT-Base": {
        },
    }

# Path to pretrained checkpoints of FAR models
FAR_MODEL_PATH = {
    "DeiT-Tiny": "",
    "DeiT-Small": "",
    "DeiT-Base": "",
    }

# Path to checkpoints of pruned FAR models
FAR_MODEL_PRUNED_PATH = {
    "DeiT-Tiny": "",
    "DeiT-Small": "",
    "DeiT-Base": "",
    }

# Path to attention-trained-only checkpoints of FAR models
FAR_MODEL_ATTN_ONLY_PATH = {
    "DeiT-Tiny": "",
    "DeiT-Small": "",
    "DeiT-Base": "",
}

# Path to models to be visualized
VIS_MODEL_PATH = {
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
        "weight":  "",
        },
    "Multihead-Tiny": {
        "weight":  "",
        },
    }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/home/u17/yuxinr/FAR/FAR"


###### ----------- Parser utils ----------- ######
def parse_replace(value):
    """Parse --replace parameter, support single numbers and ranges"""
    parts = value.split()
    numbers = []
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))
    return numbers


def get_unique_output_dir(base_dir):
    model_dir = Path(base_dir) / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    lock_file = Path(model_dir) / ".output_dir_lock"
    lock_file.touch(exist_ok=True)

    with open(lock_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = model_dir / timestamp
        while output_dir.exists():
            current_time += datetime.timedelta(seconds=1)
            timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
            output_dir = model_dir / timestamp
        
        output_dir.mkdir(parents=True, exist_ok=False)
        print(f"Created directory: {output_dir}")
        fcntl.flock(f, fcntl.LOCK_UN)

    return output_dir


def fill_default_args(args, full_arg=True):

    if args.data_path == "":
        args.data_path = DATA_PATH[args.dataset]
        print(f"Using default dataset path {args.data_path}")
    if not full_arg:
        return args
    
    args.output_dir = get_unique_output_dir(args.base_dir)

    if args.base_model_name == "":
        args.base_model_name = BASE_MODEL_PATH[args.base_model]["name"]
        print(f"Using default base model version {args.base_model_name}")

    if args.base_weight == "":
        args.base_weight = BASE_MODEL_PATH[args.base_model]["weight"]
        print(f"Using default base model weight {args.base_weight}")
    
    if args.far_weight == "":
        args.far_weight = FAR_MODEL_PATH[args.far_model]
        if args.ds_pruned:
            args.far_weight = FAR_MODEL_PRUNED_PATH[args.base_model]
        print(f"Using default pretrained FAR model weight {args.far_weight}")

    if args.vis_weight == "":
        args.vis_weight = VIS_MODEL_PATH[args.vis_model]["weight"]
        print(f"Using default visualization model weight {args.vis_weight}")

    if args.vis_model_name == "":
        args.vis_model_name = VIS_MODEL_PATH[args.vis_model]["name"]
        print(f"Using default visualization model version {args.vis_model_name}")

    if args.mode == "train" and args.ds_in_train:
        if args.base_ds_weight == "":
            args.base_ds_weight = BASE_MODEL_DS_PATH[args.base_model][args.dataset]
            print(f"Using default downstreamed base model weight {args.base_ds_model}")
        if args.skip_train_attn and args.attn_weight == "":
            args.attn_weight = FAR_MODEL_ATTN_ONLY_PATH[args.base_model]
    
    if args.mode == "prune":
        args.reg_in_train = True

    return args


###### ----------- Shared parsers for FAR modeling ----------- ######
def get_common_parser():
    parser = argparse.ArgumentParser("parser for basic environment and dataset", add_help=False)
    
    # Environment setups
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--base-dir", default=BASE_DIR, help="Base output directory")
    parser.add_argument("--output-dir", default='', help="Output path, do NOT change here")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    # parser.set_defaults(pin_mem=True)
    parser.add_argument("--random-seed", action="store_true", help="use random seed")
    parser.add_argument('--seed', default=42, type=int, help="Random seed")
    
    # Dataset parameters
    parser.add_argument("--input-size", default=224, type=int, help="expected images size for model input")
    parser.add_argument("--dataset", default="IMNET", type=str, 
                        choices=["IMNET", "CIFAR10", "CIFAR100", "INAT18", "INAT19", "FLOWER", "CAR"])
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument("--data-path", default="", type=str, help="Path of dataset")
    parser.add_argument("--nb-classes", default=1000, type=int, 
                        help="Number of classes in dataset (default:1000)")
    parser.add_argument("--train-subset", default=1.0, type=float, help="Sampling rate from dataset")
    
    # Data augment parameters
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

    return parser


def get_modeling_parser():
    parser = argparse.ArgumentParser("Parser for FAR model training process", add_help=False)
    
    # Base model setups
    parser.add_argument("--base-model", default="DeiT-Tiny", choices=["DeiT-Tiny", "DeiT-Small", "DeiT-Base"],
                        type=str, help="Name of ATTENTION based model. e.g.: DeiT-Tiny")
    parser.add_argument("--base-model-name", default="", type=str, help="base-model index name, e.g.: deit_tiny_patch16_224")
    parser.add_argument("--base-weight", default="", help="path of base-model checkpoint")
    parser.add_argument("--far-model", default="DeiT-Tiny", choices=["DeiT-Tiny", "DeiT-Small", "DeiT-Base"],
                        type=str, help="Name of attention base model which FAR model is distilled from. e.g.: DeiT-Tiny")
    parser.add_argument("--far-weight", default="", help="path of far-model checkpoint")
    parser.add_argument("--vis-model", default="DeiT-Tiny",
                        type=str, help="Name of model to be visualized")
    parser.add_argument("--vis-model-name", default="", type=str, 
                        help="vis-model index name for attention base models, e.g.: deit_tiny_patch16_224")
    parser.add_argument("--vis-weight", default="", help="path of visualization target model checkpoint")
    parser.add_argument("--attn-weight", default="", help="path of attn part pretrained replace structure")
    
    # Running mode
    parser.add_argument("--mode", default="train", choices=["train", "eval", "finetune", "downstream", "prune"], 
                        help="Runing mode")
    
    # Training setups
    parser.add_argument("--train-mode", default="sequential", choices=["parallel", "sequential"])
    parser.add_argument("--step", default=12, type=int, help="Step length when sequentially replace blocks and training")
    parser.add_argument("--interm-model", default="", type=str, help="Path of intermediate model in sequential training")
    parser.add_argument("--replace", default="0-11", type=parse_replace, help="List of indices or range of blocks to replace")
    parser.add_argument("--rep-by", default="multi-lstm", choices=["mixer", "lstm", "multi-lstm"], 
                        help="Structure used to replace attention")
    parser.add_argument("--skip-train-attn", action='store_true', 
                        help="Use pretrained attn part instead of train from scratch")
    parser.add_argument("--block-ft", action='store_true', 
                        help="Block-level finetune the replaced blocks after training attention")
    parser.set_defaults(block_ft=True)
    parser.add_argument("--reg-in-train", action='store_true', 
                        help="Adding regularization in attn training")
    parser.add_argument("--init-with-pretrained", action='store_true', 
                        help="Downstream training from deit downstream model")
    parser.add_argument("--train-loss", default="combine", choices=["similarity", "classification", "combine"],
                        type=str, help="Criterion using in training")
    
    # Training parameters
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
    parser.add_argument("--block-ft-train-loss", default="classification", choices=["classification", "combine", "similarity"],
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
    parser.add_argument("--ds-in-train", action='store_true', 
                        help="Downstream training from deit downstream model. Use TRAIN mode.")
    parser.add_argument("--base-ds-weight", default="", type=str, help="Path of downstreamed attention based model weight")
    parser.add_argument("--ds-mode", default="full", choices=["full", "FC", "FC+head"])
    parser.add_argument("--ds-pruned", action='store_true', help="Downstream on pruned model")
    
    # Finetuning setups
    parser.add_argument("--ft-mode", default="head", choices=["head", "sequential"])
    parser.add_argument("--ft-model", default="", help="Path of model to be finetuned")
    parser.add_argument("--ft-loss", default="classification", 
                        choices=["similarity", "classification", "combine"],
                        type=str, help="Criterion using in global finetune")
    parser.add_argument("--ft-batch-size", default=256, type=int, help="Batch size when global finetuning")
    parser.add_argument("--ft-epochs", default=30, type=int, help="Training epochs when global finetuning")
    parser.add_argument("--ft-lr", type=float, default=5e-6, metavar='LR', 
                        help='Learning rate when global finetuning')
    parser.add_argument('--ft-unscale-lr', action='store_true',
                        help="Not scale lr according to batch size when global finetuning")
    parser.add_argument('--ft-warmup-epochs', type=int, default=5, 
                        help='Number of warmup epochs when global finetuning')
    parser.add_argument('--ft-warmup-lr', type=float, default=1e-5,
                        help='Warm-up initial learning rate when global finetuning')
    parser.add_argument('--ft-sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler when global finetuning (default: "cosine")')
    
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
    

def get_full_parser():
    parser = argparse.ArgumentParser(
        "FAR: Attention replacement",
        parents=[get_common_parser(), get_modeling_parser()]
    )
    return parser
