import torch
import torch.nn as nn
from timm.models import create_model
import sys
sys.modules["models"] = sys.modules[__name__]

class MultiHeadLstmBlock(nn.Module):
    def __init__(self, input_dim=192, head_num=3, hidden_dim=64, num_layers=1, dropout=0.1):
        super(MultiHeadLstmBlock, self).__init__()
        
        mask_ih = get_block_mask(input_dim // head_num, hidden_dim, head_num)
        mask_hh = get_block_mask(hidden_dim, hidden_dim, head_num)
        self.input_dim = input_dim
        self.head_num = head_num
        self.output_dim = input_dim
        self.hidden_dim = hidden_dim * head_num
        self.pre_proj = nn.Linear(self.input_dim, self.input_dim)
        self.token_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, bidirectional=True,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.proj = nn.Linear(2*self.hidden_dim, self.output_dim)
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                param.data *= mask_ih
            elif 'weight_hh' in name:
                param.data *= mask_hh
        
    def forward(self, x):
        # x = self.token_norm(x) 
        x = self.pre_proj(x)
        self.pre_proj_out = x.clone()
        lstm_out, _ = self.lstm(x)
        self.lstm_out = lstm_out.clone()
        out = self.proj(lstm_out)
        return out


class ParallelLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, 
                 proj: nn.Linear, pre_proj: nn.Linear = None, token_norm: nn.LayerNorm = None, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=input_dim // num_heads,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True, dropout=dropout)
            for _ in range(num_heads)
        ])
        
        self.proj = proj
        self.token_norm = token_norm if token_norm is not None else nn.LayerNorm(input_dim)
        self.pre_proj = pre_proj if pre_proj is not None else nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x = self.token_norm(x)
        x = self.pre_proj(x)
        self.pre_proj_out = x.clone()
        chunks = torch.chunk(x, self.num_heads, dim=-1)
        
        outs_fwd, outs_rev = [], []
        for lstm, chunk in zip(self.lstms, chunks):
            y = lstm(chunk)[0]    
            outs_fwd.append(y[..., : self.hidden_dim])   # 64-f
            outs_rev.append(y[..., self.hidden_dim :])   # 64-r
        y_cat = torch.cat(outs_fwd + outs_rev, dim=-1)   # [B, N, 384]
        self.lstm_out = y_cat.clone()
        
        return self.proj(y_cat)
    

class AttnBlockWithOutput(nn.Module):
    """DeiT Block with shortcut"""
    def __init__(self, original_block):
        super(AttnBlockWithOutput, self).__init__()
        self.attn = original_block.attn
        self.mlp = original_block.mlp
        self.norm1 = original_block.norm1
        self.norm2 = original_block.norm2
        self.drop_path = getattr(original_block, "drop_path", nn.Identity()) 
        self.block_output = None

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.drop_path(self.attn(y))
        y = self.norm2(x)
        x = x + self.drop_path(self.mlp(y))
        self.block_output = x.clone()
        return x
       

def load_weight(model, weight):
    if weight.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            weight, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(weight, map_location="cpu")
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
            
    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed
    
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
    # print("Missing keys:", missing_keys)
    # print("Unexpected keys:", unexpected_keys)
    return model


def get_block_mask(input_dim, hidden_dim, head_num):
    in_per_head  = input_dim      # = 64
    hid_per_head = hidden_dim     # = 64
    H_total      = hid_per_head * head_num  # 192

    mask = torch.zeros(4*H_total, input_dim*head_num)  # 768×192
    for h in range(head_num):
        col = slice(h*in_per_head, (h+1)*in_per_head)
        for g in range(4):  # i f g o
            row = slice(g*H_total + h*hid_per_head,
                         g*H_total + (h+1)*hid_per_head)
            mask[row, col] = 1.0
    return mask

    
def replace_attention(args, model, repl_blocks, target = None, model_name = ""):
    print(f"Replacing blocks: {repl_blocks}; Replace by: {target}")
    
    for blk_index in repl_blocks:
        block = model.blocks[blk_index]
        if target == "attn":
            repl_block = AttnBlockWithOutput(block)
        elif target == "multi-lstm":
            input_dim = block.attn.qkv.in_features
            num_heads = block.attn.num_heads
            head_dim = block.attn.head_dim
            if blk_index == 0:
                print(f"Replacing setting: input_dim {input_dim}, num_heads {num_heads}, head_dim {head_dim}")
            repl_block = AttnBlockWithOutput(block)
            multi_lstm_block = MultiHeadLstmBlock(input_dim, num_heads, head_dim)
            repl_block.attn = multi_lstm_block
        else:
            raise NotImplementedError("Not available replace architecture (attn/multi-lstm)")  

        repl_block.to(args.device)
        model.blocks[blk_index] = repl_block
    return model


def set_requires_grad(model, mode = "train", target_blocks = [], target_part = "attn", trainable=True):
    if model == None:
        print("Model is None, cannot set trainable parts.")
        return
    
    raw_model = model.module if hasattr(model, "module") else model
    
    target_names = [f"blocks.{block}." for block in target_blocks]
    
    print("Trainable Params:")
    # Global fintune when transfer to downstream datasets
    if mode == "downstream":
        if target_part == "full":
            for name, param in model.named_parameters():
                param.requires_grad = trainable
        elif target_part == "FC":
            for name, param in model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    if "mlp" in name:
                        param.requires_grad = trainable
                        print(name)
            for name, param in raw_model.head.named_parameters():
                param.requires_grad = True
                print(name)
        elif target_part == "FC+head":
            for name, param in model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    if "mlp" in name:
                        param.requires_grad = trainable
                        print(name)
            for name, param in raw_model.head.named_parameters():
                param.requires_grad = True
                print(name)
        
    elif mode == "prune":
        for name, param in model.named_parameters():
            param.requires_grad = trainable
    
    elif mode == "finetune":
        # turn the classification head to trainable
        if target_part == "head":
            for param in model.parameters():
                param.requires_grad = False
            for name, param in raw_model.head.named_parameters():
                param.requires_grad = True
                print(name)
        # turn the whole blocks to trainable
        elif target_part == "sequential":
            for name, param in model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    param.requires_grad = trainable
                    print(name)
    
    elif mode == "train":
        # turn the whole block to trainable
        if target_part == "block":
            for name, param in model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    param.requires_grad = trainable
                    print(name)
        # turn the FC layers in replaced block to trainable      
        elif target_part == "FC":
            for name, param in model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    if "mlp" in name:
                        param.requires_grad = trainable
                        print(name)
        # turn the replaced MLP-Mixer to trainable
        elif target_part == "attn":
            for name, param in model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    if "attn" in name and "teacher" not in name:
                        param.requires_grad = trainable
                        print(name)
        elif target_part == "FC+head":
            for name, param in model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    if "mlp" in name:
                        param.requires_grad = trainable
                        print(name)
            for name, param in raw_model.head.named_parameters():
                param.requires_grad = True
                print(name)
                        
    else:
        raise NotImplementedError("Not available set_requires_grad mode (train/finetune/downstream)")  
                       

def load_downstream_model(model_path, args, source="local", model_name=""):
    if source == "local":
        model = torch.load(model_path)
    elif source == "online":
        model = create_model(
            model_name=model_name, pretrained=False, num_classes=args.nb_classes, drop_rate=args.drop,
            drop_path_rate=args.drop_path, drop_block_rate=None, img_size=args.input_size
            )
    
    if hasattr(model, "module"):
        model = model.module
        
    embed_dim = model.head.in_features
    out_dim = args.nb_classes
    model.head = nn.Linear(embed_dim, out_dim)
    nn.init.trunc_normal_(model.head.weight, std=0.02) 
    nn.init.zeros_(model.head.bias)
    return model
