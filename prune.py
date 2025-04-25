import re
import json
import copy
import utils
import fcntl
import torch
import models
import random
import datetime
import argparse
import numpy as np

import matplotlib
matplotlib.use("pdf")

import torch.utils.data
import torch.nn.parallel
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn

from pathlib import Path

from data import load_dataset
from train import train_model_with_reg, evaluate_model, compute_lstm_reg

from timm.utils import NativeScaler
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

DATA_PATH = {
    "IMNET": "/srv/datasets/imagenet/",
    }
PRETRAINED_PATH = {
    "Tiny": "/home/yuxinr/AttnDistill/models/2025-03-07-23-15-10/model_seq0.pth",
    "Small": "/home/yuxinr/AttnDistill/models/2025-03-17-23-15-18/model_seq0.pth",
    "Base": "",
    }
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/yuxinr/AttnDistill/"


def get_args_parser():
    parser = argparse.ArgumentParser("Prune with DeepHoyer", add_help=False)
    
    # Environment setups
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--base-dir", default=BASE_DIR, help="Base output path")
    parser.add_argument("--output-dir", default='', help="Output path")
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    
    # Model setups
    parser.add_argument("--model-path", default="", help="path of pretrained model checkpoint")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='LSTM',
                        choices=["LSTM", "Mixer"], help='pretrained model architecture')    
    parser.add_argument("--scale", default="Tiny", choices=["Tiny", "Small", "Base"],
                        type=str, metavar="MODEL")
    parser.add_argument('--reg', type=int, default=3, metavar='R',
                        help='regularization type: 0:None 1:L1 2:Hoyer 3:HS')
    parser.add_argument('--decay', type=float, default=1e-4, metavar='D',
                        help='weight decay for regularizer (default: 0.001)')  
        
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
    # data parameters
    parser.add_argument("--dataset", default="IMNET", type=str, 
                        choices=["IMNET", "CIFAR10", "CIFAR100", "INAT18", "INAT19", "FLOWER", "CAR"])
    parser.add_argument("--data-path", type=str, help="Path of dataset")
    parser.add_argument("--nb-classes", default=1000, type=int, 
                        help="Number of classes in dataset (default:1000)")
    parser.add_argument("--input-size", default=224, type=int, help="expected images size for model input")
    parser.add_argument("--train-subset", default=1.0, type=float, help="Sampling rate from dataset")
    
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

    # Training parameters
    parser.add_argument("--epochs", default=100, type=int, help="Training epochs")
    parser.add_argument("--ft-epochs", default=100, type=int, help="Fintuning epochs after pruning")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--ft-batch-size", default=256, type=int)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "sgd"')
    parser.add_argument('--ft-opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "sgd"')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument("--lr", type=float, default=5e-5, metavar='LR', help='Replaced attention learning rate')
    parser.add_argument("--ft-lr", type=float, default=5e-5, metavar='LR', help='Replaced attention learning rate')
    parser.add_argument('--lr_decay', default=0.1, type=float, metavar='LRD', help='learning rate decay')
    parser.add_argument('--lr_int', '--learning-rate-interval', default=30, type=int,
                        metavar='LRI', help='learning rate decay interval')
    parser.add_argument('--seed', default=42, type=int, help="Random seed")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')    
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    
    # Pruning parameters
    parser.add_argument('--sensitivity', type=float, default=1e-4, help="threshold used for pruning")
    
    
    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser
    

def get_unique_output_dir(base_dir):
    model_dir = Path(base_dir) / "models"
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
   

def prune(args, model, data_loader_val):
    # Initial pruning
    print("--- Pruning ---")
    actual_model = model.module if hasattr(model, "module") else model
    masks = {}
    threshold = args.sensitivity 
    if args.arch == "LSTM":
        for block in actual_model.blocks:
            weight_groups = compute_lstm_reg(block, args, "mask")
            vector_masks = {
                name: (values > threshold).float()
                for name, values in weight_groups.items()
                }
            num_layers = block.attn.lstm.num_layers
            for name, param in block.named_parameters():
                if "weight" in name and "attn" in name: # Only mask ih and hh layers in LSTM
                    match = re.search(r'_l(\d+)', name)
                    if match:
                        layer_idx = int(match.group(1))  
                    ih_name = re.sub(r'hh', 'ih', name)
                    ih_name = re.sub(r"attn.lstm.", "", ih_name)
                    last_ih_name = re.sub(r'_l(\d+)(_reverse)?', f'_l{layer_idx - 1}\\2', ih_name)
                    input_dim = block.attn.lstm.input_size
                    if "hh" in name:
                        row_mask = vector_masks[ih_name]
                        col_mask = row_mask
                    elif "ih" in name and layer_idx > 0:
                        row_mask = vector_masks[ih_name]
                        col_mask = vector_masks[last_ih_name]
                    elif "ih" in name: # First ih layer
                        row_mask = vector_masks[ih_name]
                        col_mask = torch.ones(input_dim, device=row_mask.device, dtype=row_mask.dtype)
                    elif "proj" in name:
                        last_ih_name = f"weight_ih_l{num_layers-1}"
                        col_mask = vector_masks[last_ih_name]
                        if block.attn.lstm.bidirectional:
                            re_last_ih_name = f"weight_ih_l{num_layers-1}_reverse"
                            col_mask = torch.cat([col_mask, vector_masks[re_last_ih_name]])
                        row_mask = torch.ones(input_dim, device=col_mask.device, dtype=col_mask.dtype)
                        
                    mat_mask = torch.outer(row_mask, col_mask)
                    if "proj" in name:
                        full_mask = mat_mask
                    # Repeat for 4 gates (i, f, g, o)
                    else:
                        full_mask = mat_mask.repeat(4, 1) 
                    masks[name] = full_mask
                    
                    tensor = param.data.cpu().numpy()
                    mask_tensor = full_mask.data.cpu().numpy()
                    new_param = np.where(mask_tensor == 0, 0, tensor)
                    param.data = torch.from_numpy(new_param).to(args.device)       

    else:
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                new_mask = np.where(abs(tensor) < threshold, 0, tensor)
                mask = np.where(abs(tensor) < threshold, 0., 1.)
                masks[name] = torch.from_numpy(mask).float().to(args.device)
                p.data = torch.from_numpy(new_mask).to(args.device)        

    utils.print_nonzeros(model)
    print('Pruned model evaluation...')
    evaluate_model(data_loader_val, args.device, model)
    
    return model, masks
    

def main(args):
    
    print(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=4))
    if args.model_path == "":
        print("Using default PRETRAINED_PATH")
        args.model_path = PRETRAINED_PATH[args.scale]
    if args.data_path is None:
        args.data_path = DATA_PATH[args.dataset]
    print(f"Running in Pruning mode, pruning on {args.scale} model replaced by {args.arch}")
    print(f"Using device: {args.device}")
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True
    
    data_loader_train = load_dataset(args, "train")
    data_loader_val, _ = load_dataset(args, "val")
    
    # Load model
    print(f"Loading model: {args.model_path}")
    model = torch.load(args.model_path)
    model.to(args.device)
    models.set_requires_grad(model, "prune") # whole model trainable
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable params: {n_parameters}")
    
    args.output_dir = get_unique_output_dir(args.base_dir)
    
    criterion = torch.nn.CrossEntropyLoss()
    # regularized training
    # optimizer = create_optimizer(args, model_without_ddp)
    # loss_scaler = NativeScaler()
    # lr_scheduler, _ = create_scheduler(args, optimizer)
    # reg_model, reg_model_dict = train_model_with_reg(
    #     args=args, model=model, 
    #     train_data=data_loader_train, test_data=data_loader_val,
    #     optimizer=optimizer, loss_scaler=loss_scaler, criterion=criterion,
    #     lr_scheduler=lr_scheduler, n_parameters=n_parameters                         
    #     )
    # 
    # if utils.get_rank() == 0:
    #     save_path = args.output_dir / "model_reg.pth"
    #     if hasattr(reg_model, 'module'):
    #         torch.save(reg_model.module, save_path)
    #     else:
    #         torch.save(reg_model, save_path)
    args.replace = list(range(12))
    reg_model = torch.load("/home/yuxinr/AttnDistill/models/2025-04-18-01-50-56/model_seq0.pth")
    reg_model.to(args.device)
    models.set_requires_grad(reg_model, "prune") # whole model trainable
    
    # pruning
    reg_model_without_ddp = reg_model
    if args.distributed:
        reg_model = torch.nn.parallel.DistributedDataParallel(reg_model, device_ids=[args.gpu])
        reg_model_without_ddp = reg_model.module
    pruned_model, pruning_mask = prune(args, reg_model_without_ddp, data_loader_val)

    # finetuning pruned model
    pruned_model_without_ddp = pruned_model
    if args.distributed:
        pruned_model = torch.nn.parallel.DistributedDataParallel(pruned_model, device_ids=[args.gpu])
        pruned_model_without_ddp = pruned_model.module
    brgs = copy.deepcopy(args)
    brgs.lr = args.ft_lr
    brgs.epochs = args.ft_epochs
    brgs.opt = args.ft_opt
    brgs.batch_size = args.ft_batch_size
    
    optimizer = create_optimizer(brgs, pruned_model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(brgs, optimizer)
    
    fted_model, ft_model_dict = train_model_with_reg(
        args=brgs, model=pruned_model, 
        train_data=data_loader_train, test_data=data_loader_val,
        optimizer=optimizer, loss_scaler=loss_scaler, criterion=criterion,
        lr_scheduler=lr_scheduler, n_parameters=n_parameters,
        mode="finetune", mask=pruning_mask
        )
    
    if utils.get_rank() == 0:
        save_path = args.output_dir / "model_ft.pth"
        if hasattr(fted_model, 'module'):
            torch.save(fted_model.module, save_path)
        else:
            torch.save(fted_model, save_path)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Prune with DeepHoyer", parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    main(args)
    