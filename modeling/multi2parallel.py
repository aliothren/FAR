import json
import torch
import config
from pathlib import Path
from copy import deepcopy
from data import load_dataset
from train import evaluate_model
from architectures import ParallelLSTM, set_requires_grad
 

def split_block_diag_weight(W, num_heads=3):
    """
    W: [4*H*num_heads,  D]  or  [4*H*num_heads, 4*H*num_heads]
    return list[num_heads], each [4*H,  D/num_heads] or [4*H, 4*H]
    """
    out_dim, in_dim = W.shape
    H = out_dim // (4 * num_heads)
    in_blk = in_dim // num_heads
    parts = []
    for i in range(num_heads):
        r0, r1 = i * 4 * H, (i + 1) * 4 * H
        c0, c1 = i * in_blk, (i + 1) * in_blk
        parts.append(W[r0:r1, c0:c1].clone())
    return parts


def split_block_diag_bias(b, num_heads=3):
    """
    b: [4*H*num_heads]  → 3 * [4*H]
    """
    H = b.shape[0] // (4 * num_heads)
    return [b[i*4*H:(i+1)*4*H].clone() for i in range(num_heads)]


def convert_block(multi_blk: torch.nn.Module):
    """multi_blk is MultiHeadLstmBlock instance"""
    D          = multi_blk.input_dim
    H_per_head = multi_blk.hidden_dim // multi_blk.head_num
    H_heads    = multi_blk.head_num

    # create new parallel LSTM , share pre_proj / token_norm / proj
    para_blk = ParallelLSTM(
        input_dim=D,
        hidden_dim=H_per_head,
        num_heads=H_heads,
        proj        = deepcopy(multi_blk.proj),
        pre_proj    = deepcopy(multi_blk.pre_proj),
        token_norm  = deepcopy(multi_blk.token_norm),
        dropout     = 0.0,      
    )

    def w_slice(W, head_idx, H_per_head, head_num):
        H_total = H_per_head * head_num
        cols = slice(head_idx * H_per_head, (head_idx + 1) * H_per_head)    
        parts = []
        for g in range(4):
            row_start = g * H_total + head_idx * H_per_head
            row_end   = row_start + H_per_head
            parts.append(W[row_start:row_end, cols])    
        return torch.cat(parts, dim=0)    
        
    def b_slice(b, head_idx, H_per_head, head_num):
        H_total = H_per_head * head_num
        parts = []
        for g in range(4):                           # i, f, g, o
            start = g * H_total + head_idx * H_per_head
            end   = start + H_per_head
            parts.append(b[start:end])
        return torch.cat(parts, dim=0).clone()   
 
    multi_lstm  = multi_blk.lstm
    for i, single_lstm in enumerate(para_blk.lstms):
        # 正向
        single_lstm.weight_ih_l0.data.copy_(w_slice(multi_lstm.weight_ih_l0, i, H_per_head, H_heads))
        single_lstm.weight_hh_l0.data.copy_(w_slice(multi_lstm.weight_hh_l0, i, H_per_head, H_heads))
        single_lstm.bias_ih_l0 .data.copy_(b_slice(multi_lstm.bias_ih_l0,  i, H_per_head, H_heads))
        single_lstm.bias_hh_l0 .data.copy_(b_slice(multi_lstm.bias_hh_l0,  i, H_per_head, H_heads))
        # 反向
        single_lstm.weight_ih_l0_reverse.data.copy_(w_slice(multi_lstm.weight_ih_l0_reverse, i, H_per_head, H_heads))
        single_lstm.weight_hh_l0_reverse.data.copy_(w_slice(multi_lstm.weight_hh_l0_reverse, i, H_per_head, H_heads))
        single_lstm.bias_ih_l0_reverse .data.copy_(b_slice(multi_lstm.bias_ih_l0_reverse,  i, H_per_head, H_heads))
        single_lstm.bias_hh_l0_reverse .data.copy_(b_slice(multi_lstm.bias_hh_l0_reverse,  i, H_per_head, H_heads))

    return para_blk


def assert_block_equal(big_blk, para_blk, tol=1e-5, device='cpu'):
    big_blk, para_blk = big_blk.to(device), para_blk.to(device)
    big_blk.eval()
    para_blk.eval()
    x = torch.randn(2, 197, big_blk.input_dim, device=device)
    with torch.no_grad():
        y_big  = big_blk(x)
        y_para = para_blk(x)
        pre_big = big_blk.pre_proj_out
        pre_para = para_blk.pre_proj_out
        lstm_big = big_blk.lstm_out
        lstm_para = para_blk.lstm_out
    diff_pre = (pre_big - pre_para).abs().max().item()
    diff_lstm = (lstm_big[..., 0:64] - lstm_para[..., 0:64]).abs().max().item()
    diff = (y_big - y_para).abs().max().item()
    print(f"diff_pre={diff_pre}")
    print(f"diff_lstm={diff_lstm}, diff={diff}")
    # exit(0)
    assert diff < tol, f"block diff={diff}"
    return diff


if __name__ == '__main__':
    # Modify multihead model and target parallel model path here
    multi_path = Path("/home/yuxinr/far/FAR/modeling/checkpoints/2025-06-03-14-45-48/model_block_seq0.pth")
    save_path = multi_path.with_name("model_parallel.pth")
    parser = config.get_common_parser()
    args = parser.parse_args()
    args = config.fill_default_args(args, full_arg=False)
    print(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=4))

    # Load dataset
    data_loader_val, dataset_val = load_dataset(args, "val")

    # Load multihead lstm model
    multi_model = torch.load(multi_path, map_location="cpu")

    # Evaluate multihead lstm model
    multi_model.eval()   
    set_requires_grad(multi_model, target_blocks=[])
    multi_model.to(args.device)
    test_stats = evaluate_model(data_loader_val, args.device, multi_model)
    print(f"Accuracy on multihead LSTM: {test_stats['acc1']:.1f}%")
    multi_model.to("cpu")

    # Convert multihead LSTM into parallel LSTM structure
    for i, blk in enumerate(multi_model.blocks):
        multi_blk_attn = deepcopy(blk.attn)
        blk.attn = convert_block(blk.attn)
        assert_block_equal(multi_blk_attn, blk.attn, device="cpu")
        print(f"✓ block {i:2d} passed equivalence test")

    torch.save(multi_model, save_path)

    print(f"Evaluating model {save_path}")
    multi_model.to(args.device)
    test_stats = evaluate_model(data_loader_val, args.device, multi_model)
    print(f"Accuracy on parallel LSTM: {test_stats['acc1']:.1f}%")
