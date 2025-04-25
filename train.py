import sys
import math
import time
import json
import torch
import utils
import models
import datetime
import numpy as np

from typing import Iterable
from timm.utils import accuracy
from loss import CosineSimilarityLoss


@torch.no_grad()
def evaluate_model(data_loader, device, model, teacher_model = None, loss_type="classification"):
    actual_model = model.module if hasattr(model, "module") else model
    actual_teacher = teacher_model.module if (teacher_model is not None and hasattr(teacher_model, "module")) else teacher_model
    autocast_ctx = torch.amp.autocast(device_type="cuda") if device.type == "cuda" else torch.cpu.amp.autocast(device_type="cpu")
    if loss_type == "classification":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_type == "similarity":
        criterion = CosineSimilarityLoss()
    else:
        raise ValueError("Invalid evaluation loss type (classification/similarity).") 

    # switch to evaluation mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for images, target in metric_logger.log_every(data_loader, 100, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if loss_type == "classification":
            with autocast_ctx:
                output = model(images)
                loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            metric_logger.update(test_class_loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        elif loss_type == "similarity":
            with autocast_ctx:
                output = actual_model.forward_features(images)
                teacher_output = actual_teacher.forward_features(images)
                loss = criterion(output, teacher_output)
            metric_logger.update(test_cosine_loss=loss.item())
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if loss_type == "classification":
        print("Classification loss on test set:")
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.test_class_loss))
    elif loss_type == "similarity":
        print("Similarity loss on test set:")
        print(metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def apply_masks(model, multi_lstm=False, lstm_mask={}, reg=False, reg_mask={}):
    if multi_lstm:
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'weight_ih' in name:
                    param.grad.data.mul_(lstm_mask["mask_ih"].to(param.device))
                elif 'weight_hh' in name:
                    param.grad.data.mul_(lstm_mask["mask_ih"].to(param.device))
    if reg:
        for name, param in model.named_parameters():
            if 'weight' in name and name in reg_mask:
                param.grad.data.mul_(reg_mask[name].to(param.device))
                
                
def build_grad_mask_fn(model, multi_lstm=False, lstm_mask=None, reg=False, reg_mask=None):
    def grad_mask_fn():
        apply_masks(
            model=model,
            multi_lstm=multi_lstm,
            lstm_mask=lstm_mask or {},
            reg=reg,
            reg_mask=reg_mask or {}
        )
    return grad_mask_fn  
     
    
def train_one_epoch(args, mode, model: torch.nn.Module, teacher_model: torch.nn.Module, 
                    replace: list, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0):
    actual_model = model.module if hasattr(model, "module") else model
    actual_teacher = teacher_model.module if (teacher_model is not None and hasattr(teacher_model, "module")) else teacher_model
    autocast_ctx = torch.autocast(device_type=device.type)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
            
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
          
        with autocast_ctx:
            if mode == "similarity":
                criterion = CosineSimilarityLoss()
                output_student = model(samples)
                output_teacher = teacher_model(samples)
                loss = 0
                for blk in replace:
                    output_block_s = actual_model.blocks[blk].block_output
                    output_block_t = actual_teacher.blocks[blk].block_output
                    loss += criterion(output_block_s, output_block_t)
                
            elif mode == "classification":
                criterion = torch.nn.CrossEntropyLoss()
                output = model(samples)
                loss = criterion(output, targets)
                
            elif mode == "combine":
                ce_criterion = torch.nn.CrossEntropyLoss()
                cos_criterion = CosineSimilarityLoss()
                output_student = model(samples)
                output_teacher = teacher_model(samples)
                loss = ce_criterion(output_student, targets)
                for blk in replace:
                    output_block_s = actual_model.blocks[blk].block_output
                    output_block_t = actual_teacher.blocks[blk].block_output
                    loss += cos_criterion(output_block_s, output_block_t)
                
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        
def compute_lstm_reg(block, args, mode="reg"):
    lstm = block.attn.lstm
    proj = block.attn.proj
    block_reg = 0.0
    num_layers = lstm.num_layers
    hidden_dim = lstm.hidden_size
    num_directions = 2 if lstm.bidirectional else 1
    weight_groups = {}
    for direction in range(num_directions):
        for layer in range(num_layers):
            # Get all related matrices
            suffix = f'_l{layer}' + ('_reverse' if direction == 1 else '')
            ih = getattr(lstm, f'weight_ih{suffix}', None)
            hh = getattr(lstm, f'weight_hh{suffix}', None)
            next_param = None
            if layer + 1 < num_layers:
                next_suffix = f'_l{layer + 1}' + ('_reverse' if direction == 1 else '')
                next_param = getattr(lstm, f'weight_ih{next_suffix}', None)
            else: # For the last layer, related to proj
                next_param = getattr(proj, 'weight', None)
                start, end = direction * hidden_dim, (direction + 1) * hidden_dim
                next_param = next_param[:, start:end]
            if ih is None or hh is None:
                continue
            
            # Reshape and concat all related channel
            # ih: [4H, input_dim] → [4, H, input_dim] → [H, 4, input_dim] → [H, 4*input_dim]
            ih_rows = ih.view(4, hidden_dim, -1).transpose(0, 1).reshape(hidden_dim, -1)
            # hh: [4H, input_dim] → [4, H, H]
            hh_split = hh.view(4, hidden_dim, hidden_dim)
            # hh_row: [4, H, H] → [H, 4, H] → [H, 4*H]
            hh_rows = hh_split.transpose(0, 1).reshape(hidden_dim, -1)
            hh_cols = hh_split.permute(2, 0, 1).reshape(hidden_dim, -1)
            # next_ih for layers and proj for the last layer
            if layer == num_layers - 1:
                next_col = next_param.transpose(0, 1)
            else:
                next_col = next_param.view(4, hidden_dim, -1).permute(2, 0, 1).reshape(hidden_dim, -1)
            
            # Calculate reg 
            weight_group = torch.cat([ih_rows, hh_rows, hh_cols, next_col], dim=1)    
            if args.reg == 1:
                reg = torch.sum(torch.norm(weight_group, p=2, dim=1))
            elif args.reg == 2:
                reg = torch.sum(torch.norm(weight_group, p=2, dim=1)) / (torch.norm(weight_group, p=2) + 1e-6)
            elif args.reg == 3:
                reg = torch.sum(torch.norm(weight_group, p=2, dim=1)) ** 2 / (torch.sum(weight_group ** 2) + 1e-6)
            else:
                reg = 0.0
            
            if mode == "reg":
                block_reg += reg
            elif mode == "mask":
                weight_groups[f'weight_ih{suffix}'] = torch.sum(weight_group, dim=1)
    
    if mode == "reg":
        return block_reg
    elif mode == "mask":
        return weight_groups
        

def train_one_epoch_with_reg(
    args, loss_mode, model: torch.nn.Module, teacher_model: torch.nn.Module, 
    replace: list, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
    device: torch.device, epoch: int, curves, step,
    loss_scaler, max_norm: float = 0, mode="reg", mask={}):
    model.train()
    actual_model = model.module if hasattr(model, "module") else model
    actual_teacher = teacher_model.module if (teacher_model is not None and hasattr(teacher_model, "module")) else teacher_model
    autocast_ctx = torch.autocast(device_type=device.type)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    if args.distributed:
        mask = {("module." + k if not k.startswith("module.") else k): v for k, v in mask.items()}

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
         
         # Calculate training loss
        with autocast_ctx:
            if loss_mode == "similarity":
                criterion = CosineSimilarityLoss()
                output_student = model(samples)
                output_teacher = teacher_model(samples)
                loss = 0
                for blk in replace:
                    output_block_s = actual_model.blocks[blk].block_output
                    output_block_t = actual_teacher.blocks[blk].block_output
                    loss += criterion(output_block_s, output_block_t)
                
            elif loss_mode == "classification":
                criterion = torch.nn.CrossEntropyLoss()
                output_student = model(samples)
                loss = criterion(output_student, targets)
                
            elif loss_mode == "combine":
                ce_criterion = torch.nn.CrossEntropyLoss()
                cos_criterion = CosineSimilarityLoss()
                output_student = model(samples)
                output_teacher = teacher_model(samples)
                loss = ce_criterion(output_student, targets)
                for blk in replace:
                    output_block_s = actual_model.blocks[blk].block_output
                    output_block_t = actual_teacher.blocks[blk].block_output
                    loss += cos_criterion(output_block_s, output_block_t)

        # Calculate reg loss
        reg = 0.0
        if mode == "reg":
            if args.arch == "LSTM" and args.decay:
                for block in actual_model.blocks:
                    reg += compute_lstm_reg(block, args, mode)
                    
            elif args.arch == "Mixer":
                if args.decay:
                    for name, param in model.named_parameters():
                        if param.requires_grad and len(list(param.size()))>1 and 'weight' in name and torch.sum(torch.abs(param))>0:
                            if args.reg==1:    
                                reg += torch.sum(torch.sqrt(torch.sum(param**2,0)))+torch.sum(torch.sqrt(torch.sum(param**2,1)))
                            elif args.reg==2:
                                reg += (torch.sum(torch.sqrt(torch.sum(param**2,0)))+torch.sum(torch.sqrt(torch.sum(param**2,1))))/torch.sqrt(torch.sum(param**2))
                            elif args.reg==3:
                                reg += ( (torch.sum(torch.sqrt(torch.sum(param**2,0)))**2) + (torch.sum(torch.sqrt(torch.sum(param**2,1)))**2) )/torch.sum(param**2)    
                            else:
                                reg = 0.0     
            else:
                raise ValueError("Invalid architechture (LSTM/Mixer).") 
                    
            total_loss = loss + args.decay*reg
            loss_value = loss.item()
            total_loss_value = total_loss.item()
        elif mode == "finetune":
            for name, param in model.named_parameters():
                if 'weight' in name and name in mask:
                    param.grad.data.mul_(mask[name].to(param.device))
            total_loss = loss
            loss_value = loss.item()
            total_loss_value = total_loss.item()
            
        if not math.isfinite(total_loss_value):
            print("Loss is {}, stopping training".format(total_loss_value))
            sys.exit(1)
        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(total_loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        torch.cuda.synchronize()

        # Update logs
        metric_logger.update(total_loss=total_loss_value)
        metric_logger.update(loss=loss_value)
        metric_logger.update(reg=reg.item() if isinstance(reg, torch.Tensor) else reg)
        prec1, prec5 = reg_accuracy(output_student, targets, topk=(1, 5))
        metric_logger.update(prec1=prec1[0])
        metric_logger.update(prec5=prec5[0])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if i and i % args.print_freq == 0:
            nonzero = total = 0
            filter_count = filter_total = 0
            total_sparsity = total_layer = 0
            for name, param in model.named_parameters():
                if 'weight' in name and len(list(param.size()))>1:
                    tensor = param.data.cpu().numpy()
                    threshold = args.sensitivity
                    new_mask = np.where(abs(tensor) < threshold, 0, tensor)
                    #p.data = torch.from_numpy(new_mask).to(device)
                    tensor = np.abs(new_mask)
                    nz_count = np.count_nonzero(tensor)
                    total_params = np.prod(tensor.shape)
                    nonzero += nz_count
                    total += total_params
                    
                    if len(tensor.shape)==4:
                        dim0 = np.sum(np.sum(tensor, axis=0),axis=(1,2))
                        dim1 = np.sum(np.sum(tensor, axis=1),axis=(1,2))
                        nz_count0 = np.count_nonzero(dim0)
                        nz_count1 = np.count_nonzero(dim1)
                        filter_count += nz_count0*nz_count1
                        filter_total += len(dim0)*len(dim1)
                        total_sparsity += 1-(nz_count0*nz_count1)/(len(dim0)*len(dim1))
                        total_layer += 1
                    if len(tensor.shape)==2:
                        dim0 = np.sum(tensor, axis=0)
                        dim1 = np.sum(tensor, axis=1)
                        nz_count0 = np.count_nonzero(dim0)
                        nz_count1 = np.count_nonzero(dim1)
                        filter_count += nz_count0*nz_count1
                        filter_total += len(dim0)*len(dim1)
                        total_sparsity += 1-(nz_count0*nz_count1)/(len(dim0)*len(dim1))
                        total_layer += 1
                    
            elt_sparsity = (total-nonzero)/total
            input_sparsity = (filter_total-filter_count)/filter_total
            output_sparsity = total_sparsity/total_layer
            
            curves[step, 0] = len(data_loader)*epoch+i
            curves[step, 1] = metric_logger.meters["loss"].avg
            curves[step, 2] = reg
            curves[step, 3] = elt_sparsity
            curves[step, 4] = input_sparsity
            curves[step, 5] = output_sparsity
            curves[step, 6] = metric_logger.meters["total_loss"].avg
            step += 1          
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, curves, step

  
def train_model(
    args, stage, loss_mode, model, teacher_model, train_data, test_data,
    optimizer, loss_scaler, lr_scheduler, n_parameters,
    reg_mode="reg", mask = {}):
    
    actual_model = model.module if hasattr(model, "module") else model

    if utils.get_rank() == 0:
        with (args.output_dir / "log.txt").open("a") as f:
            f.write("Args: " + str(args) + "\n")
        print(f"Start {stage} training for {args.epochs} epochs, regularization status: {args.reg_in_train}")      
            
    checkpoint_path = args.output_dir / f"{stage}_checkpoint.pth"
    best_checkpoint_path = args.output_dir / f"{stage}_best_checkpoint.pth"
    
    if teacher_model is not None:
        teacher_model.eval()
    
    if args.rep_by == "multi-lstm":
        input_dim = actual_model.blocks[0].attn.input_dim
        head_num = actual_model.blocks[0].attn.head_num
        hidden_dim = actual_model.blocks[0].attn.hidden_dim // head_num
        mask_ih = models.get_block_mask(input_dim // head_num, hidden_dim, head_num)
        mask_hh = models.get_block_mask(hidden_dim, hidden_dim, head_num)
        grad_mask_fn = build_grad_mask_fn(
                            model,
                            multi_lstm=True,
                            lstm_mask={"mask_ih": mask_ih, "mask_hh": mask_hh}
                        )
        loss_scaler = utils.CustomNativeScaler(grad_mask_fn=grad_mask_fn)
    
    start_time = time.time()
    max_accuracy = 0.0
    if args.reg_in_train:
        # columns: [step, loss, reg, elt_sparsity, input_sparsity, output_sparsity, total_loss]
        curves = np.zeros((args.epochs*(len(train_data)//10),7))
        valid = np.zeros((args.epochs,3))
        step = 0  
    for epoch in range(args.epochs):
        # Set only target part in training mode
        actual_model.eval()
        for blk in args.replace:
            if stage == "attn":
                actual_model.blocks[blk].attn.train()
            elif stage in ["block", "global"]:
                actual_model.blocks[blk].train()
            else:
                raise ValueError("Invalid training stage (attn/block/global).") 
            
        if args.distributed:
            train_data.sampler.set_epoch(epoch)
    
        if args.reg_in_train:
            train_stats, curves, step = train_one_epoch_with_reg(
                args, loss_mode=loss_mode, model=model, teacher_model=teacher_model, 
                replace=args.replace, data_loader=train_data, 
                optimizer=optimizer, device=args.device, epoch=epoch, curves=curves, step=step,
                loss_scaler=loss_scaler, max_norm=args.clip_grad, mode=reg_mode, mask=mask)
        else:
            train_stats = train_one_epoch(
                args=args, mode=loss_mode, model=model, teacher_model=teacher_model, 
                replace=args.replace, data_loader=train_data, 
                optimizer=optimizer, device=args.device, epoch=epoch, 
                loss_scaler=loss_scaler, max_norm=args.clip_grad)
         
        with torch.no_grad():
            num_zeros = (actual_model.blocks[0].attn.lstm.weight_ih_l0 == 0).sum().item()
            total_elements = actual_model.blocks[0].attn.lstm.weight_ih_l0.numel()
            zero_ratio = num_zeros / total_elements
            print(f"[MASK CHECK] weight_ih_l0 zero ratio: {zero_ratio:.4f} ({num_zeros}/{total_elements})")   
               
        lr_scheduler.step(epoch)
        
        # Evaluate training
        test_stats_class = evaluate_model(test_data, args.device, model, None, "classification") # Classification result on test set
        if loss_mode in ["similarity", "combine"]:
            test_stats_cosine = evaluate_model(test_data, args.device, model, teacher_model, "similarity") # Similarity loss on test set

        model_dict = {
            'model': actual_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
            }
        # Save checkpoint and logs
        if utils.get_rank() == 0:
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save(model_dict, checkpoint_path)
            # Save best checkpoint
            if max_accuracy < test_stats_class["acc1"]:
                max_accuracy = test_stats_class["acc1"]
                torch.save(model_dict, best_checkpoint_path)
            print(f'Max accuracy: {max_accuracy:.2f}%')
            # Save logs
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats_class.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            if loss_mode in ["similarity", "combine"]:
                log_stats.update({f'test_cosine_{k}': v for k, v in test_stats_cosine.items()})
            with (args.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if args.reg_in_train:
                test_acc = [test_stats_class["acc1"], test_stats_class["acc5"]]
                utils.save_train_fig(
                    save_path=args.output_dir, epoch=epoch, acc=test_acc,
                    curves=curves, valid=valid, step=step, mode=stage
                )

    if utils.get_rank() == 0:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{stage} finished.")
        print('Training time {}'.format(total_time_str))
        with (args.output_dir / "log.txt").open("a") as f:
            f.write("Training time:" + json.dumps(total_time_str) + "\n")
            
    return model, model_dict

    
def train_model_with_reg(
    args, model, train_data, test_data,
    optimizer, loss_scaler, criterion, lr_scheduler, n_parameters,
    mode="reg", mask = {}):
    
    actual_model = model.module if hasattr(model, "module") else model
    actual_model.train()
    
    if utils.get_rank() == 0:
        with (args.output_dir / "log.txt").open("a") as f:
            f.write("Args: " + str(args) + "\n")
    checkpoint_path = args.output_dir / f"{mode}_checkpoint.pth"
    best_checkpoint_path = args.output_dir / f"{mode}_best_checkpoint.pth"
    if utils.get_rank() == 0:
        print(f"Start {mode} training for {args.epochs} epochs")      

    start_time = time.time()
    max_accuracy = 0.0
    # columns: [step, loss, reg, elt_sparsity, input_sparsity, output_sparsity, total_loss]
    curves = np.zeros((args.epochs*(len(train_data)//10),7))
    valid = np.zeros((args.epochs,3))
    step = 0
    
    for epoch in range(args.epochs):
        if args.distributed:
            train_data.sampler.set_epoch(epoch)
            
        train_stats, curves, step = train_one_epoch_with_reg(
            args, loss_mode="classification", model=model, teacher_model=None,
            replace=args.replace, data_loader=train_data, optimizer=optimizer, 
            device=args.device, epoch=epoch, curves=curves, step=step,
            loss_scaler=loss_scaler, max_norm=args.clip_grad, mode=mode, mask=mask)

        lr_scheduler.step(epoch)
        
        # Evaluate training
        test_stats = evaluate_model(test_data, args.device, model)

        model_dict = {
            'model': actual_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
            }
        # Save checkpoint and logs
        if utils.get_rank() == 0:
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save(model_dict, checkpoint_path)
            # Save best checkpoint
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                torch.save(model_dict, best_checkpoint_path)
            print(f'Max accuracy: {max_accuracy:.2f}%')
            # Save logs
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            with (args.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            # Save training figures
            test_acc = [test_stats["acc1"], test_stats["acc5"]]
            utils.save_train_fig(
                save_path=args.output_dir, epoch=epoch, acc=test_acc,
                curves=curves, valid=valid, step=step, mode=mode
            )

    if utils.get_rank() == 0:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Train with regularization finished.")
        print('Training time {}'.format(total_time_str))
        with (args.output_dir / "log.txt").open("a") as f:
            f.write("Training time:" + json.dumps(total_time_str) + "\n")
            
    return model, model_dict


def reg_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
