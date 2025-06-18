import sys
import math
import time
import json
import torch
import utils
import datetime
import numpy as np
import architectures

from typing import Iterable
from timm.utils import accuracy
from loss import CosineSimilarityLoss


def apply_masks(model, multi_lstm=False, lstm_mask={}, reg=False, reg_mask={}):
    if multi_lstm:
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'weight_ih' in name:
                    param.grad.data.mul_(lstm_mask["mask_ih"].to(param.device))
                elif 'weight_hh' in name:
                    param.grad.data.mul_(lstm_mask["mask_ih"].to(param.device))
                elif 'head_proj.weight' in name:
                    try:
                        param.grad.data.mul_(lstm_mask["mask_head"].to(param.device))
                    except:
                        print("head_proj weight shape", model.blocks[0].attn.head_proj.weight.shape)  
                        print("head_proj grad shape", model.blocks[0].attn.head_proj.weight.grad.shape)  
                        print("head_proj mask shape", lstm_mask["mask_head"].shape)  
                        exit(0)
                    
    if reg:
        for name, param in model.named_parameters():
            if "module" in name:
                name = name[len("module."):]
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
     

def compute_lstm_reg_multihead(block, args, mode="reg"):
    """
    Returns: torch.Tensor (mode="reg") or dict[str, torch.Tensor] (mode="mask")
    """
    lstm = block.attn.lstm
    proj = block.attn.proj
    # get params
    try:
        heads = block.attn.head_num
    except:
        heads = 3 # tiny
    in_dim_total = block.attn.input_dim
    hid_total = block.attn.hidden_dim  
    hid_per_head = hid_total // heads
    in_per_head = in_dim_total // heads
    num_directions = 2 if lstm.bidirectional else 1

    block_reg = 0.0
    weight_groups = {}

    for direction in range(num_directions):
        suffix = f"_l0" + ("_reverse" if direction == 1 else "")
        ih = getattr(lstm, f"weight_ih{suffix}", None)     # [4*Htot , In]
        hh = getattr(lstm, f"weight_hh{suffix}", None)     # [4*Htot , Htot]

        proj_weight = proj.weight                      # [Out , 2*Htot]
        dir_offset = direction * hid_total           # 0 或 hid_total

        for h in range(heads):
            in_start  = h * in_per_head
            in_end  = (h + 1) * in_per_head
            hid_start = h * hid_per_head
            hid_end = (h + 1) * hid_per_head

            ih_slice = ih[4*hid_start : 4*hid_end, in_start:in_end]               # [4*hid_h , in_h]
            hh_slice = hh[4*hid_start : 4*hid_end, hid_start:hid_end]             # [4*hid_h , hid_h]
            proj_slice = proj_weight[:, dir_offset + hid_start : dir_offset + hid_end]  # [Out , hid_h]

            # Reshape and concat all related channel
            ih_rows = ih_slice.view(4, hid_per_head, -1).transpose(0, 1).reshape(hid_per_head, -1)
            hh_split = hh_slice.view(4, hid_per_head, hid_per_head)
            hh_rows  = hh_split.transpose(0, 1).reshape(hid_per_head, -1)               # [Hh , 4*Hh]
            hh_cols  = hh_split.permute(2, 0, 1).reshape(hid_per_head, -1)              # [Hh , 4*Hh]
            proj_col = proj_slice.t()                                         # [Hh , Out]

            weight_group = torch.cat([ih_rows, hh_rows, hh_cols, proj_col], dim=1)
            # weight_group = torch.cat([ih_rows, hh_rows], dim=1)

            # Calculate reg 
            if args.reg == 1:
                reg = torch.sum(torch.norm(weight_group, p=2, dim=1))
            elif args.reg == 2:
                reg = (torch.sum(torch.norm(weight_group, p=2, dim=1))
                           / (torch.norm(weight_group, p=2) + 1e-6))
            elif args.reg == 3:
                reg = (torch.sum(torch.norm(weight_group, p=2, dim=1)) ** 2
                           / (torch.sum(weight_group ** 2) + 1e-6))
            else:
                reg = 0.0

            if mode == "reg":
                block_reg += reg
            elif mode == "mask":
                key = f"weight_ih{suffix}"
                if key not in weight_groups:
                    weight_groups[key] = torch.zeros(hid_total, device=weight_group.device)
                weight_groups[key][hid_start:hid_end] = torch.sum(weight_group, dim=1)

    if mode == "reg":
        return block_reg
    elif mode == "mask":
        return weight_groups
 

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


def train_one_epoch(
    args, loss_mode, model: torch.nn.Module, teacher_model: torch.nn.Module, 
    replace: list, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
    device: torch.device, epoch: int, curves, step, loss_scaler, 
    max_norm: float = 0, add_reg=False):
    
    model.train()
    
    actual_model = model.module if hasattr(model, "module") else model
    actual_teacher = teacher_model.module if (teacher_model is not None and hasattr(teacher_model, "module")) else teacher_model
    autocast_ctx = torch.autocast(device_type=device.type)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
            
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
                    # output_block_s = actual_model.blocks[blk].block_output
                    # output_block_t = actual_teacher.blocks[blk].block_output
                    output_block_s = actual_model.blocks[blk].attn_output
                    output_block_t = actual_teacher.blocks[blk].attn_output
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
        cal_reg = False
        if add_reg or cal_reg:
            if args.rep_by == "multi-lstm" and args.decay:
                for block in actual_model.blocks:
                    reg += compute_lstm_reg_multihead(block, args, mode="reg")
        if add_reg:
            total_loss = loss + args.decay*reg
        else:
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
    args, stage, # stage: ["attn", "block", "global", "reg", "reg_ft"]
    loss_mode, model, teacher_model, train_data, test_data,
    optimizer, lr_scheduler, n_parameters, mask = {}):
    
    actual_model = model.module if hasattr(model, "module") else model
    if teacher_model is not None:
        teacher_model.eval()

    if utils.get_rank() == 0:
        with (args.output_dir / "log.txt").open("a") as f:
            f.write("Args: " + str(args) + "\n")
        print(f"Start {stage} training for {args.epochs} epochs, regularization status: {args.reg_in_train}")      
            
    checkpoint_path = args.output_dir / f"{stage}_checkpoint.pth"
    best_checkpoint_path = args.output_dir / f"{stage}_best_checkpoint.pth"
    
    # Generate custom scaler
    use_multi_lstm_mask = False
    use_reg_mask = False
    lstm_mask = {}
    reg_mask = {}
    # Use lstm mask when replace by multihead LSTMs, to imitate parallel LSTMs
    if args.rep_by == "multi-lstm":
        input_dim = actual_model.blocks[0].attn.input_dim
        head_num = actual_model.blocks[0].attn.head_num
        hidden_dim = actual_model.blocks[0].attn.hidden_dim // head_num

        mask_ih = architectures.get_block_mask(input_dim // head_num, hidden_dim, head_num)
        mask_hh = architectures.get_block_mask(hidden_dim, hidden_dim, head_num)
        mask_head = architectures.get_head_mask(hidden_dim, head_num)
        use_multi_lstm_mask = True
        lstm_mask={"mask_ih": mask_ih, "mask_hh": mask_hh, "mask_head": mask_head}
        print("Using multi-head LSTM masks.")
    # Use regularization mask in finetune after pruning
    if args.reg_in_train and mask:
        use_reg_mask = True
        reg_mask = mask
        print("Using pruning masks.")
    grad_mask_fn = build_grad_mask_fn(
                        model,
                        multi_lstm=use_multi_lstm_mask,
                        lstm_mask=lstm_mask,
                        reg=use_reg_mask,
                        reg_mask=reg_mask
                    )
    loss_scaler = utils.CustomNativeScaler(grad_mask_fn=grad_mask_fn)
    
    start_time = time.time()
    max_accuracy = 0.0
    # Initialize plot tools
    # columns: [step, loss, reg, elt_sparsity, input_sparsity, output_sparsity, total_loss]
    curves = np.zeros((args.epochs*(len(train_data)//10),7))
    valid = np.zeros((args.epochs,3))
    step = 0  
    
    # Train model
    for epoch in range(args.epochs):
        if args.distributed:
            train_data.sampler.set_epoch(epoch)
        actual_model.train()
        
        add_reg = True if (args.reg_in_train and not mask) else False
        train_stats, curves, step = train_one_epoch( 
            args, loss_mode=loss_mode, model=model, teacher_model=teacher_model, 
            replace=args.replace, data_loader=train_data, optimizer=optimizer,
            device=args.device, epoch=epoch, curves=curves, step=step,
            loss_scaler=loss_scaler, max_norm=args.clip_grad, add_reg=add_reg
        )
         
        with torch.no_grad():
            num_zeros_wih = (actual_model.blocks[0].attn.lstm.weight_ih_l0 == 0).sum().item()
            total_elements_wih = actual_model.blocks[0].attn.lstm.weight_ih_l0.numel()
            zero_ratio_wih = num_zeros_wih / total_elements_wih
            num_zeros_proj = (actual_model.blocks[0].attn.head_proj.weight == 0).sum().item()
            total_elements_proj = actual_model.blocks[0].attn.head_proj.weight.numel()
            zero_ratio_proj = num_zeros_proj / total_elements_proj
            print(f"[MASK CHECK]\n \
                    weight_ih_l0 zero ratio: {zero_ratio_wih:.4f} ({num_zeros_wih}/{total_elements_wih})\n \
                    head_proj zero ratio: {zero_ratio_proj:.4f} ({num_zeros_proj}/{total_elements_proj})  ")   
               
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
