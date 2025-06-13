import io
import os
import time
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist

from pathlib import Path
from collections import defaultdict, deque
from timm.utils.agc import adaptive_clip_grad


def dispatch_clip_grad(parameters, value: float, mode: str = 'norm', norm_type: float = 2.0):
    """ Dispatch to gradient clipping method

    Args:
        parameters (Iterable): model parameters to clip
        value (float): clipping value/factor/norm, mode dependant
        mode (str): clipping mode, one of 'norm', 'value', 'agc'
        norm_type (float): p-norm, default 2.0
    """
    if mode == 'norm':
        torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
    elif mode == 'value':
        torch.nn.utils.clip_grad_value_(parameters, value)
    elif mode == 'agc':
        adaptive_clip_grad(parameters, value, norm_type=norm_type)
    else:
        assert False, f"Unknown clip mode ({mode})."


# Modify NativeScaler from timms, add grad_mask_fn
class CustomNativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self, grad_mask_fn=None, device='cuda'):
        self.grad_mask_fn = grad_mask_fn
        try:
            self._scaler = torch.amp.GradScaler(device=device)
        except (AttributeError, TypeError) as e:
            self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            clip_mode='norm',
            parameters=None,
            create_graph=False,
            need_update=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        
        if self.grad_mask_fn is not None:
            self.grad_mask_fn()
            
        if need_update:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    print(f"Using distributed mode, rank={args.rank}, GPU={args.gpu}")

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def save_train_fig(save_path, epoch, acc, curves, valid, step, mode):
    
    # evaluate on validation set
    valid[epoch, 0] = epoch
    valid[epoch, 1], valid[epoch, 2] = acc
    
    save_path = Path(save_path) / "figs"
    save_path.mkdir(parents=True, exist_ok=True)
    # plot training curve
    np.savetxt(save_path / f'curves_{mode}.dat', curves)
    
    clr1 = (0.5, 0., 0.)
    clr2 = (0.0, 0.5, 0.)
    fig, ax1 = plt.subplots()
    fig2, ax3 = plt.subplots()
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()
    ax1.set_xlabel('steps')
    ax1.set_ylabel('Loss', color=clr1)
    ax1.tick_params(axis='y', colors=clr1)
    ax2.set_ylabel('Total loss', color=clr2)
    ax2.tick_params(axis='y', colors=clr2)
    
    ax3.set_xlabel('steps')
    ax3.set_ylabel('Sparsity', color=clr1)
    ax3.tick_params(axis='y', colors=clr1)
    ax4.set_ylabel('Reg', color=clr2)
    #ax4.yscale('log')
    ax4.tick_params(axis='y', colors=clr2)
    
    start = 0
    end = step
    markersize = 12
    coef = 2.
    ax1.plot(curves[start:end, 0], curves[start:end, 1], '--', color=[c*coef for c in clr1], markersize=markersize)
    ax2.plot(curves[start:end, 0], curves[start:end, 6], '-', color=[c*coef for c in clr2], markersize=markersize)
    ax3.plot(curves[start:end, 0], curves[start:end, 3], '--', color=[c*1. for c in clr1], markersize=markersize)
    ax3.plot(curves[start:end, 0], curves[start:end, 4], '-', color=[c*1.5 for c in clr1], markersize=markersize)
    ax3.plot(curves[start:end, 0], curves[start:end, 5], '-', color=[c*2. for c in clr1], markersize=markersize)
    ax4.plot(curves[start:end, 0], curves[start:end, 2], '-', color=[c*coef for c in clr2], markersize=markersize)
    
    #ax2.set_ylim(bottom=20, top=100)
    ax1.legend(('Train loss'), loc='lower right')
    ax2.legend(('Total loss'), loc='lower left')
    fig.savefig(save_path / f'loss-vs-steps_{mode}.png')
    
    #ax4.set_ylim(bottom=20, top=100)
    ax3.legend(('Elt_sparsity','Filter_sparsity','Average_sparsity'), loc='lower right')
    ax4.legend(('Reg'), loc='lower left')
    fig2.savefig(save_path / f'sparsity-vs-steps_{mode}.png')
    
    # plot validation curve
    np.savetxt(save_path / f'valid_{mode}.dat', valid)
    
    fig3, ax5 = plt.subplots()
    ax6 = ax5.twinx()
    ax5.set_xlabel('epochs')
    ax5.set_ylabel('Acc@1', color=clr1)
    ax5.tick_params(axis='y', colors=clr1)
    ax6.set_ylabel('Acc@5', color=clr2)
    ax6.tick_params(axis='y', colors=clr2)
    
    start = 0
    end = epoch+1
    markersize = 12
    coef = 2.
    ax5.plot(valid[start:end, 0], valid[start:end, 1], '--', color=[c*coef for c in clr1], markersize=markersize)
    ax6.plot(valid[start:end, 0], valid[start:end, 2], '-', color=[c*coef for c in clr2], markersize=markersize)
    
    #ax2.set_ylim(bottom=20, top=100)
    ax5.legend(('Acc@1'), loc='lower right')
    ax6.legend(('Acc@5'), loc='lower left')
    fig3.savefig(save_path / f'accuracy-vs-epochs_{mode}.png')
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)

    
def print_nonzeros(model):
    nz_param = 0
    nz_channel = 0
    total = 0
    for name, p in model.named_parameters():
        if 'weight' in name:
            # Element-wise sparsity
            tensor = p.data.cpu().numpy()
            if tensor.ndim < 2:
                print(f"{name:20} | Skipped channel-wise sparsity (not a matrix): shape = {tensor.shape}")
                continue
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nz_param += nz_count
            total += total_params
            print(f'{name:20} |')
            print(f'Element: nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
            
            # Channel-wise sparsity
            abs_tensor = np.abs(tensor)
            if abs_tensor.ndim == 4:
                reshaped = abs_tensor.reshape(abs_tensor.shape[0], -1)  # [out_channels, in_channels × kh × kw]
            elif tensor.ndim == 2:
                reshaped = abs_tensor
            num_rows, num_cols = reshaped.shape
            num_nz_rows = np.count_nonzero(np.sum(reshaped, axis=1) > 0)
            num_nz_cols = np.count_nonzero(np.sum(reshaped, axis=0) > 0)
            effective_params = num_nz_rows * num_nz_cols
            nz_channel += effective_params
            print(f'Channel: nz_rows = {num_nz_rows:7} / {num_rows:7} ({100 * num_nz_rows / num_rows:6.2f}%)')
            print(f'Channel: nz_cols = {num_nz_cols:7} / {num_cols:7} ({100 * num_nz_cols / num_cols:6.2f}%)')
            print(f'Channel: total_pruned = {total_params - effective_params :7} ({100 * effective_params / total_params}% left) | shape = {tensor.shape}')

    print(f'alive: {nz_channel}, pruned : {total - nz_channel}, total: {total}, Compression rate : {total/nz_channel:10.2f}x  ({100 * (total-nz_channel) / total:6.2f}% pruned)')
