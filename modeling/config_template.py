import torch
import fcntl
import argparse
import datetime
from pathlib import Path

###### ----------- Global PATHs and configs ----------- ######

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = ""

# Path to datasets
DATA_PATH = {
    "IMNET": "",
    "CIFAR100": "",
    "CIFAR10": "",
    "INAT18": "",
    "INAT19": "",
    "FLOWER": "",
    "CAR": "",
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
    "DeiT-Base-384": {
        "name": "deit_base_patch16_384",
        "weight": "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
    },
    "AViT-Tiny": {
        "name": "avit_tiny_patch16_224",
        "weight": "",
        # "weight": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    },
    "AViT-Small": {
        "name": "avit_small_patch16_224",
        # "weight": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        "weight": "",
    },
    "AViT-Base": {
        "name": "avit_base_patch16_224",
        "weight": "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
    },
}

# Path to downstreamed checkpoints of ATTENTION architecture models
BASE_MODEL_DS_PATH = {
    "DeiT-Tiny": {
        "CIFAR10": "",
        "CIFAR100": "",
        "INAT18": "",
        "INAT19": "",
        "FLOWER": "",
        "CAR": "",
    },
    "DeiT-Small": {
        "CIFAR10": "",
        "CIFAR100": "",
        "FLOWER": "",
        "CAR": "",
        "INAT18": "",
        "INAT19": "",
    },
    "DeiT-Base": {
        "CIFAR10": "",
        "CIFAR100": "",
        "FLOWER": "",
        "CAR": "",
        "INAT18": "",
        "INAT19": "",
    },
}

# Path to pretrained checkpoints of FAR models
FAR_MODEL_PATH = {
    "LSTM-Tiny": "",
    "LSTM-Small": "",
    "LSTM-Base": "",
    "Mamba-Tiny": "",
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
    "DeiT-Base-384": "",
    "AViT-Tiny": "",
    "AViT-Small": "",
    "AViT-Base": "",
}

# Path to concated FAR models with parallel trained blocks
FAR_MODEL_CONCAT_PATH = {
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
    "Multihead-Tiny": {
        "name": "Multihead-Tiny",
        "weight": "",
    },
    "Mamba-Tiny": {
        "name": "Mamba-Tiny",
        "weight": "",
    },
    "AViT-attn": {
        "name": "avit_tiny_patch16_224",
        "weight": "",
    },
    "AViT-lstm": {
        "name": "AViT-lstm",
        "weight": "",
    },
    "AViT-mamba": {
        "name": "AViT-mamba",
        "weight": "",
    },
    
}

# Path to models for statistics
STAT_MODEL_PATH = {
    "DeiT-Tiny": {
        "name": "deit_tiny_patch16_224",
        "weight": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    },
    "DeiT-Base": {
        "name": "deit_base_patch16_224",
        "weight": "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
    },
    "DeiT-Base-384": {
        "name": "deit_base_patch16_384",
        "weight": "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
    },
    "Multihead-Tiny": {
        "name": "Multihead-Tiny",
        "weight": "",
    },
    "Mamba-Tiny": {
        "name": "Mamba-Tiny",
        "weight": "",
    },
    "Mamba-Base": {
        "name": "Mamba-Base",
        "weight": "",
    },
    "Mamba-Base-384": {
        "name": "Mamba-Base-384",
        "weight": "",
    },
    "AViT-attn": {
        "name": "avit_tiny_patch16_224",
        "weight": "",
    },
    "AViT-lstm": {
        "name": "AViT-lstm",
        "weight": "",
    },
    "AViT-mamba": {
        "name": "AViT-mamba",
        "weight": "",
    },
    
}

###### ----------- Parser utils ----------- ######
def parse_replace(value):
    """Parse --replace parameter, support single numbers and ranges"""
    parts = value.split()
    numbers = []
    for part in parts:
        if "-" in part:
            start, end = map(int, part.split("-"))
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

    if args.stat_weight == "":
        args.stat_weight = STAT_MODEL_PATH[args.stat_model]["weight"]
        print(f"Using default stat model weight {args.stat_weight}")

    if args.stat_model_name == "":
        args.stat_model_name = STAT_MODEL_PATH[args.stat_model]["name"]
        print(f"Using default stat model version {args.stat_model_name}")

    if args.mode == "train":
        if args.base_ds_weight == "" and args.ds_in_train:
            args.base_ds_weight = BASE_MODEL_DS_PATH[args.base_model][args.dataset]
            print(f"Using default downstreamed base model weight {args.base_ds_weight}")
        if args.skip_train_attn and args.attn_weight == "":
            args.attn_weight = FAR_MODEL_ATTN_ONLY_PATH[args.base_model]
        if args.use_concat and args.concat_weight == "":
            args.concat_weight = FAR_MODEL_CONCAT_PATH[args.base_model]

    if args.mode == "prune":
        args.reg_in_train = True

    return args


###### ----------- Shared parsers for FAR modeling ----------- ######
def get_common_parser():
    parser = argparse.ArgumentParser(
        "parser for basic environment and dataset", add_help=False
    )

    # Environment setups
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--base-dir", default=BASE_DIR, help="Base output directory")
    parser.add_argument(
        "--output-dir", default="", help="Output path, do NOT change here"
    )
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--pin-mem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument(
        "--fixed-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fixed seed (default: True)",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")

    # Dataset parameters
    parser.add_argument(
        "--input-size",
        default=224,
        type=int,
        help="expected images size for model input",
    )
    parser.add_argument(
        "--dataset",
        default="IMNET",
        type=str,
        choices=["IMNET", "CIFAR10", "CIFAR100", "INAT18", "INAT19", "FLOWER", "CAR"],
    )
    parser.add_argument(
        "--inat-category",
        default="name",
        choices=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "supercategory",
            "family",
            "genus",
            "name",
        ],
        type=str,
        help="semantic granularity",
    )
    parser.add_argument("--data-path", default="", type=str, help="Path of dataset")
    parser.add_argument(
        "--nb-classes",
        default=1000,
        type=int,
        help="Number of classes in dataset (default:1000)",
    )
    parser.add_argument(
        "--train-subset", default=1.0, type=float, help="Sampling rate from dataset"
    )

    # Data augment parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.3,
        metavar="PCT",
        help="Color jitter factor (default: 0.3)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)',
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )
    parser.add_argument(
        "--eval-crop-ratio", default=0.875, type=float, help="Crop ratio for evaluation"
    )
    parser.add_argument(
        "--repeated-aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="used in distributed training",
    )

    # Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )

    return parser


def get_modeling_parser():
    parser = argparse.ArgumentParser(
        "Parser for FAR model training process", add_help=False
    )

    # Base model setups
    parser.add_argument(
        "--base-model",
        default="DeiT-Tiny",
        choices=[
            "DeiT-Tiny",
            "DeiT-Small", 
            "DeiT-Base", 
            "DeiT-Base-384", 
            "AViT-Tiny", 
            "AViT-Small", 
            "AViT-Base",
        ],
        type=str,
        help="Name of ATTENTION based model. e.g.: DeiT-Tiny",
    )
    parser.add_argument(
        "--base-model-name",
        default="",
        type=str,
        help="base-model index name, e.g.: deit_tiny_patch16_224",
    )
    parser.add_argument(
        "--base-weight", default="", help="path of base-model checkpoint"
    )
    parser.add_argument(
        "--far-model",
        default="LSTM-Tiny",
        type=str,
        help="Name of attention base model which FAR model is distilled from. e.g.: DeiT-Tiny",
    )
    parser.add_argument("--far-weight", default="", help="path of far-model checkpoint")
    parser.add_argument(
        "--vis-model",
        default="DeiT-Tiny",
        type=str,
        help="Name of model to be visualized",
    )
    parser.add_argument(
        "--vis-mode",
        default="token",
        choices=["token", "avit"],
        help="Choose to visualize token-similarity or avit token sparsity",
    )
    parser.add_argument(
        "--vis-model-name",
        default="",
        type=str,
        help="vis-model index name for attention base models, e.g.: deit_tiny_patch16_224",
    )
    parser.add_argument(
        "--vis-weight", default="", help="path of visualization target model checkpoint"
    )

    parser.add_argument(
        "--stat-model",
        default="DeiT-Tiny",
        type=str,
        help="Name of model for statistics",
    )
    parser.add_argument(
        "--stat-mode",
        default="sparsity",
        choices=["sparsity", "throughput"],
        help="Choose statistics mode",
    )
    parser.add_argument(
        "--stat-model-name",
        default="",
        type=str,
        help="stat-model index name for attention base models, e.g.: deit_tiny_patch16_224",)
    parser.add_argument(
        "--stat-weight", default="", help="path of target model checkpoint for statistics"
    )
    parser.add_argument(
        "--attn-weight",
        default="",
        help="path of attn part pretrained replace structure",
    )
    parser.add_argument(
        "--concat-weight",
        default="",
        help="path of blockly pretrained and concated weight",
    )

    # Running mode
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval", "finetune", "downstream", "prune"],
        help="Runing mode",
    )

    # Training setups
    parser.add_argument(
        "--train-mode", default="sequential", choices=["parallel", "sequential"]
    )
    parser.add_argument(
        "--step",
        default=12,
        type=int,
        help="Step length when sequentially replace blocks and training",
    )
    parser.add_argument(
        "--interm-model",
        default="",
        type=str,
        help="Path of intermediate model in sequential training",
    )
    parser.add_argument(
        "--replace",
        default="0-11",
        type=parse_replace,
        help="List of indices or range of blocks to replace",
    )
    parser.add_argument(
        "--rep-by",
        default="multi-lstm",
        choices=["mixer", "lstm", "multi-lstm", "avit", "mamba"],
        help="Structure used to replace attention",
    )
    parser.add_argument(
        "--skip-train-attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use pretrained attn part instead of train from scratch",
    )
    parser.add_argument(
        "--use-concat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use concated pretrained blocks instead of train from scratch",
    )
    parser.add_argument(
        "--block-ft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Block-level finetune the replaced blocks after training attention",
    )
    parser.add_argument(
        "--reg-in-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Adding regularization in attn training",
    )
    parser.add_argument(
        "--init-with-pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downstream training from deit downstream model",
    )
    parser.add_argument(
        "--train-loss",
        default="combine",
        choices=["similarity", "classification", "combine"],
        type=str,
        help="Criterion using in training",
    )

    # Augment parameters
    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--model-ema", 
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Exponential Moving Average to update model",
    )
    parser.add_argument("--model-ema-decay", type=float, default=0.99996, help='')
    parser.add_argument(
        "--model-ema-force-cpu",
        action=argparse.BooleanOptionalAction,
        default=False, 
        help=''
    )
    parser.add_argument(
        '--smoothing', 
        type=float, 
        default=0.1, 
        help='Label smoothing (default: 0.1)'
    )
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--mixup-active', type=bool, default=False, help='')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Training parameters
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=200, type=int, help="Training epochs")
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="Replaced attention learning rate",
    )
    parser.add_argument(
        "--unscale-lr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Not scale lr according to batch size",
    )
    parser.add_argument(
        "--min-lr", 
        type=float, 
        default=1e-6, 
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument(
        "--warmup-epochs", type=int, default=5, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--warmup-lr", type=float, default=1e-5, help="Warm-up initial learning rate"
    )
    parser.add_argument(
        "--decay-epochs", 
        type=float, 
        default=30, 
        metavar='N',
        help='epoch interval to decay LR'
        )
    parser.add_argument(
        '--cooldown-epochs', 
        type=int, 
        default=10, 
        metavar='N',
        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument(
        '--patience-epochs', 
        type=int, 
        default=10, 
        metavar='N',
        help='patience epochs for Plateau LR scheduler (default: 10'
        )
    parser.add_argument(
        '--decay-rate', 
        '--dr', 
        type=float, 
        default=0.1, 
        metavar='RATE',
        help='LR decay rate (default: 0.1)'
        )
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )

    parser.add_argument(
        "--block-ft-mode",
        default="full",
        choices=["block", "FC", "FC+head", "full"],
        type=str,
        help="Finetune scope in blockwise training",
    )
    parser.add_argument(
        "--block-ft-train-loss",
        default="classification",
        choices=["classification", "combine", "similarity"],
        type=str,
        help="Criterion using in blockwise training",
    )
    parser.add_argument(
        "--block-ft-batch-size",
        default=256,
        type=int,
        help="Batch size when block-level finetuning",
    )
    parser.add_argument(
        "--block-ft-epochs",
        default=100,
        type=int,
        help="Training epochs when block-level finetuning",
    )
    parser.add_argument(
        "--block-ft-lr",
        type=float,
        default=5e-5,
        metavar="LR",
        help="Learning rate when block-level finetuning",
    )
    parser.add_argument(
        "--block-ft-unscale-lr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Not scale lr according to batch size when block-level finetuning",
    )
    parser.add_argument(
        "--block-ft-warmup-epochs",
        type=int,
        default=5,
        help="Number of warmup epochs when block-level finetuning",
    )
    parser.add_argument(
        "--block-ft-warmup-lr",
        type=float,
        default=1e-6,
        help="Warm-up initial learning rate when block-level finetuning",
    )
    parser.add_argument(
        "--block-ft-sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler when block-level finetuning (default: "cosine")',
    )
    parser.add_argument(
        "--finetune-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only use for sequential train/finetune: finetune classification head after sequential train",
    )
    parser.add_argument(
        "--lora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use lora in block finetune",
    )

    # Evaluation setups
    parser.add_argument(
        "--eval-model", default="", help="Path of model to be evaluated"
    )

    # Downstream setups
    parser.add_argument(
        "--ds-in-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Downstream training from deit downstream model. Use TRAIN mode.",
    )
    parser.add_argument(
        "--base-ds-weight",
        default="",
        type=str,
        help="Path of downstreamed attention based model weight",
    )
    parser.add_argument("--ds-mode", default="full", choices=["full", "FC", "FC+head"])
    parser.add_argument(
        "--ds-pruned",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Downstream on pruned model",
    )

    # Post pruning Finetuning setups
    parser.add_argument(
        "--prune-ft-batch-size",
        default=256,
        type=int,
        help="Batch size when global finetuning",
    )
    parser.add_argument(
        "--prune-ft-epochs",
        default=100,
        type=int,
        help="Training epochs when global finetuning",
    )
    parser.add_argument(
        "--prune-ft-lr",
        type=float,
        default=5e-6,
        metavar="LR",
        help="Learning rate when global finetuning",
    )

    # Distributed training parameters
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enabling distributed training",
    )
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    # pruning setups
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="LSTM",
        choices=["LSTM", "Mixer"],
        help="pretrained model architecture",
    )
    parser.add_argument(
        "--reg",
        type=int,
        default=3,
        metavar="R",
        help="regularization type: 0:None 1:L1 2:Hoyer 3:HS",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=5e-5,
        metavar="D",
        help="weight decay for regularizer (default: 0.001)",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=100,
        type=int,
        metavar="N",
        help="print frequency (default: 100)",
    )
    parser.add_argument(
        "--sensitivity", type=float, default=1e-4, help="threshold used for pruning"
    )

    # AViT setups
    parser.add_argument(
        "--avit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use AViT architecture",
    )
    parser.add_argument(
        '--ponder_token_scale', 
        default=0.0005, 
        type=float, 
        help="scaling for ponder cost, default 0.0005 as in paper, " \
        "use 0.0 to disable ponder cost and only use distributional prior"
        )
    parser.add_argument(
        '--act_mode', 
        default=4, 
        type=int,
        help='4-token act, make sure this is always 4, ' \
        'other modes are only used for initial method comparison and exploration'
    )
    parser.add_argument(
        '--gate_scale', 
        default=10., 
        type=float, 
        help="constant for token control gate rescale"
    )
    parser.add_argument(
        '--gate_center', 
        default= 30., 
        type=float, 
        help="constant for token control gate re-center, negatived when applied"
    )
    parser.add_argument(
        '--distr_prior_alpha',
        default=0.001,
        type=float, 
        help="scaling for kl of distributional prior"
    )

    # loss weights
    parser.add_argument(
        '--sim-loss-weight', 
        default=1.0, 
        type=float, 
        help="similarity loss weight during training, default 1.0"
    )
    parser.add_argument(
        '--cls-loss-weight', 
        default=1.0, 
        type=float, 
        help="classification loss weight during training, default 1.0"
    )
    parser.add_argument(
        '--avit-loss-weight', 
        default=0.1, 
        type=float, 
        help="halting loss weight during training, default 0.1"
    )

    return parser


def get_full_parser():
    parser = argparse.ArgumentParser(
        "FAR: Attention replacement",
        parents=[get_common_parser(), get_modeling_parser()],
    )
    return parser
