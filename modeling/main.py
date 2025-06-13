import gc
import copy
import json
import torch
import utils
import random
import config
import architectures

import numpy as np
import torch.backends.cudnn as cudnn

from data import load_dataset
from train import train_model, evaluate_model

from prune import prune
from torchinfo import summary
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from peft import LoraConfig, get_peft_model, TaskType


def evaluate(args):
    print(f"Running in eval mode, args.mode: {args.mode}")
    print(f"Using device: {args.device}")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True
    # Load dataset
    data_loader_val, dataset_val = load_dataset(args, "val")
    # Load model
    print(f"Evaluation model: {args.eval_model}")
    model = torch.load(args.eval_model)
    model.to(args.device)
    
    architectures.set_requires_grad(model, target_blocks=[])
    test_stats = evaluate_model(data_loader_val, args.device, model)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    
    del model, dataset_val
    gc.collect()
    torch.cuda.empty_cache()


def train(args, seq=0):
    print(f"Running in train mode, args.mode: {args.mode}")
    print(f"Using device: {args.device}")

    if args.random_seed:
        random.seed(seed)
    else:
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Using seed: {seed}")
    
    cudnn.benchmark = True

    data_loader_train = load_dataset(args, "train")
    data_loader_val, _ = load_dataset(args, "val")
           
    # Load base models
    if args.ds_in_train:
        print(f"Training on downstream task {args.dataset}")
        print(f"Creating Base model: {args.base_model}")
        teacher_model = create_model(
            model_name=args.base_model_name, pretrained=False, num_classes=args.nb_classes, drop_rate=args.drop,
            drop_path_rate=args.drop_path, drop_block_rate=None, img_size=args.input_size
            )
        teacher_model = architectures.replace_attention(
            args=args, model=teacher_model, repl_blocks=args.replace, target="attn", 
            model_name=args.base_model
        )
        teacher_model = architectures.load_weight(teacher_model, args.base_ds_weight)
        args.init_with_pretrained = True
        if args.init_with_pretrained:
            student_model = copy.deepcopy(teacher_model)
            imnet_model = architectures.load_downstream_model(args.far_weight, args)
            for blk_idx in args.replace:
                student_model.blocks[blk_idx].attn = imnet_model.blocks[blk_idx].attn
            del imnet_model
        else:
            student_model = copy.deepcopy(teacher_model)
            student_model = architectures.replace_attention(
                args=args, model=student_model, repl_blocks=args.replace, target=args.rep_by, 
                model_name=args.base_model
            )
            
        teacher_model.to(args.device)   
        student_model.to(args.device)   
        
    else:
        print(f"Creating Base model: {args.base_model}")
        base_model = create_model(
            model_name=args.base_model_name, pretrained=False, num_classes=args.nb_classes, drop_rate=args.drop,
            drop_path_rate=args.drop_path, drop_block_rate=None, img_size=args.input_size
            )
        base_model = architectures.load_weight(base_model, args.base_weight)
        base_model.to(args.device)
    
        # Load and modify teacher model
        if args.train_loss == "classification" and args.block_ft_train_loss == "classification":
            teacher_model = None
        else:
            teacher_model = copy.deepcopy(base_model)
            teacher_model.to(args.device)
            teacher_model = architectures.replace_attention(
                args=args, model=teacher_model, repl_blocks=args.replace, target="attn", 
                model_name=args.base_model
            )
    
        # Load and modify student model
        if seq == 0:
            if args.skip_train_attn:
                student_model = torch.load(args.attn_weight)
            else:        
                student_model = architectures.replace_attention(
                    args=args, model=base_model, repl_blocks=args.replace, target=args.rep_by, 
                    model_name=args.base_model
                )
        else:
            student_model = torch.load(args.interm_model)
            student_model = architectures.replace_attention(
                args=args, model=student_model, repl_blocks=args.replace, target=args.rep_by, 
                model_name=args.base_model
            )
    # DDP wrap
    student_model.to(args.device)
    student_model_without_ddp = student_model
    if args.distributed:
        student_model = torch.nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.gpu], find_unused_parameters=True)
        student_model_without_ddp = student_model.module
    try:
        summary(student_model_without_ddp, depth=4, input_size=(1, 3, 224, 224))   
    except: 
        print("Unable to print model structure.")
        
    # Train attention part
    if not args.skip_train_attn:
        # Set trainable parameters
        print(f"Set teacher_model to trainable, blocks [], part attn")
        architectures.set_requires_grad(teacher_model, "train", [], "attn") # No trainable param in teacher
        print(f"Set student_model to trainable, blocks {args.replace}, part attn")
        architectures.set_requires_grad(student_model, "train", args.replace, "attn") # Target attn part trainable
        n_parameters = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        print(f"number of trainable params: {n_parameters}")
    
        # Set training configurations
        if not args.unscale_lr:
            linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
            args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, student_model_without_ddp)
        lr_scheduler, _ = create_scheduler(args, optimizer)

        # Train model
        trained_model, trained_model_dict = train_model(
            args=args, stage="attn", loss_mode=args.train_loss,
            model=student_model, teacher_model=teacher_model,
            train_data=data_loader_train, test_data=data_loader_val,
            optimizer=optimizer, lr_scheduler=lr_scheduler, n_parameters=n_parameters,
            )
        
    else: 
        if hasattr(student_model, 'module'):
            trained_model = student_model.module
        else:
            trained_model = student_model
    
    # Save model
    if utils.get_rank() == 0:
        save_path = args.output_dir / f"model_seq{seq}.pth"
        if hasattr(trained_model, 'module'):
            torch.save(trained_model.module, save_path)
        else:
            torch.save(trained_model, save_path)
        args.interm_model = save_path
    
    if args.block_ft:
        trained_model_without_ddp = trained_model
        if hasattr(trained_model_without_ddp, 'module'):
            trained_model_without_ddp = trained_model_without_ddp.module
        if args.distributed:
            trained_model = torch.nn.parallel.DistributedDataParallel(trained_model_without_ddp, device_ids=[args.gpu])
        
        # Prune model if args.reg_in_train    
        if args.reg_in_train:
            trained_model, pruning_mask = prune(args, trained_model_without_ddp, data_loader_val)
        else:
            pruning_mask = {}
        print(f"Doing block-level finetuning of attention-trained model...")
        
        # Continue training on trained_model
        print(f"Set trained_model to trainable, blocks {args.replace}, part {args.block_ft_mode}")
        architectures.set_requires_grad(trained_model, "train", args.replace, args.block_ft_mode) # Target part trainable
        print(f"Set teacher_model to trainable, blocks [], part {args.block_ft_mode}")
        architectures.set_requires_grad(teacher_model, "train", [], "block") # Not trainable
            
        n_parameters = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
        print(f"number of trainable params: {n_parameters}")
        
        # Set block-level parameters as training params
        args.batch_size = args.block_ft_batch_size
        args.epochs = args.block_ft_epochs
        args.lr = args.block_ft_lr
        if not args.block_ft_unscale_lr:
            linear_scaled_lr = args.lr * args.batch_size / 512.0
            args.lr = linear_scaled_lr
        args.warmup_epochs = args.block_ft_warmup_epochs
        args.warmup_lr = args.block_ft_warmup_lr
        args.sched = args.block_ft_sched
        
        # Set training configurations
        optimizer = create_optimizer(args, trained_model_without_ddp)
        lr_scheduler, _ = create_scheduler(args, optimizer)
        
        # Finetune model
        fted_model, fted_model_dict = train_model(
            args=args, stage="block", loss_mode=args.block_ft_train_loss,
            model=trained_model, teacher_model=teacher_model,
            train_data=data_loader_train, test_data=data_loader_val,
            optimizer=optimizer, lr_scheduler=lr_scheduler, 
            n_parameters=n_parameters, mask=pruning_mask
            )
        
        # Save finetuned model
        save_path = args.output_dir / f"model_block_seq{seq}.pth"
        args.interm_model = save_path
        if utils.get_rank() == 0:
            save_path = args.output_dir / f"model_block_seq{seq}.pth"
            if hasattr(fted_model, 'module'):
                torch.save(fted_model.module, save_path)
            else:
                torch.save(fted_model, save_path)
        
        del optimizer, fted_model, teacher_model, data_loader_train, data_loader_val
        gc.collect()
        torch.cuda.empty_cache()


def downstream(args, pretrained_path):
    print(f"Running in downstream mode, args.mode: {args.mode}")
    print(f"Using device: {args.device}")

    if args.random_seed:
        random.seed(seed)
    else:
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Using seed: {seed}")
    
    cudnn.benchmark = True

    data_loader_train = load_dataset(args, "train")
    data_loader_val, _ = load_dataset(args, "val")
    
    args.train_loss = "classification"
    print(f"Loading pretrained model: {pretrained_path} with class num {args.nb_classes}")
    model = architectures.load_downstream_model(pretrained_path, args)
    model.to(args.device)
    architectures.set_requires_grad(
        model, mode="downstream", target_blocks=list(range(12)), target_part=args.ds_mode
        )
    
    if args.ds_pruned:
        args.reg_in_train = True
        args.sensitivity = 0
        model, pruning_mask = prune(args, model, data_loader_val)
    else:    
        pruning_mask = {}
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
            )
        model_without_ddp = model.module
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable params: {n_parameters}")
            
    # Set training configurations
    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)    
    fted_model, fted_model_dict = train_model(
        args=args, stage="global", loss_mode=args.train_loss,
        model=model, teacher_model=None,
        train_data=data_loader_train, test_data=data_loader_val,
        optimizer=optimizer, lr_scheduler=lr_scheduler, n_parameters=n_parameters,
        mask=pruning_mask
        )
    
    if utils.get_rank() == 0:
        save_path = args.output_dir / "model_ds.pth"
        if hasattr(fted_model, 'module'):
            torch.save(fted_model.module, save_path)
        else:
            torch.save(fted_model, save_path)
    

if __name__ == '__main__':
    parser = config.get_full_parser()
    args = parser.parse_args()
    args = config.fill_default_args(args)

    if utils.get_rank() == 0:
        print(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=4))
    utils.init_distributed_mode(args)
    
    if args.mode == "eval":
        evaluate(args)
        
    elif args.mode == "train":
        # Sequentially replace blocks and train, args.step blocks each time
        if args.train_mode == "sequential":
            seq = 0
            print(f"Sequentially training blocks {args.replace}, {args.step} blocks once...")
            blocks = args.replace.copy()
            while len(blocks) > 0:
                brgs = copy.deepcopy(args) # Copy args to avoid change on args 
                brgs.replace = blocks[: brgs.step] # Modify brgs for this seq training
                
                train(args=brgs, seq=seq)
                
                # Deliver params for next seq
                args.output_dir = brgs.output_dir
                args.interm_model = brgs.interm_model
                # blocks = blocks[args.step :]
                blocks=[]
                seq += 1
                
        # Replace and train all blocks in args.replace in once 
        elif args.train_mode == "parallel":
            print(f"Parallel training blocks {args.replace}...")
            train(args)
        
        else:
            raise ValueError("Invalid train_mode (sequential/parallel).") 
        
    elif args.mode == "downstream": 
        args.replace = list(range(12))
        downstream(args, args.far_weight)
           
    else:
        raise ValueError("Invalid mode (eval/train/finetune/downstream).") 
        