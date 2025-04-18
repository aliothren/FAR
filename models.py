import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
param_dict_mixer = {
    "DeiT-Tiny":{
        "num_patches": 197,
        "token_hid_dim": 384,
    },
    "DeiT-Small":{
        "num_patches": 197,
        "token_hid_dim": 384,
    },
    "DeiT-Base":{ # to be determined
        "num_patches": 197,
        "token_hid_dim": 3072,
    }
}
param_dict_lstm = {
    "DeiT-Tiny":{
        "input_dim": 192,
        "hidden_dim": 128,
        "output_dim": 192,
        "num_layers": 1,
    },
    "DeiT-Small":{
        "input_dim": 384,
        "hidden_dim": 256,
        "output_dim": 384,
        "num_layers": 1,
    },
    "DeiT-Base":{
        "input_dim": 768,
        "hidden_dim": 512,
        "output_dim": 768,
        "num_layers": 1,
    }
}


class MlpBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, dropout = 0.):
        super(MlpBlock, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        y = self.fc1(x)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return y


class MixerBlock(nn.Module):
    """This is the mixer block to replace attention module"""
    def __init__(self, model_name, dropout = 0. ):
        super(MixerBlock, self).__init__()
        
        self.param_dict = param_dict_mixer
        self.token_mixing = MlpBlock(self.param_dict[model_name]["num_patches"],
                                     self.param_dict[model_name]["token_hid_dim"],
                                     dropout)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.token_mixing(x)
        x = x.transpose(1, 2)
        return x


class LstmBlock(nn.Module):
    def __init__(self, model_name, num_layers=1, dropout=0.1):
        super(LstmBlock, self).__init__()
        self.param_dict = param_dict_lstm
        input_dim = self.param_dict[model_name]["input_dim"]
        output_dim = self.param_dict[model_name]["output_dim"]
        hidden_dim = self.param_dict[model_name]["hidden_dim"]
        num_layers = self.param_dict[model_name]["num_layers"]
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.proj = nn.Linear(2*hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.proj(lstm_out)
        return out


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

    
def replace_attention(model, repl_blocks, target = None, model_name = ""):
    print(f"Replacing blocks: {repl_blocks}; Replace by: {target}")
    
    for blk_index in repl_blocks:
        block = model.blocks[blk_index]
        if target == "attn":
            repl_block = AttnBlockWithOutput(block)
        elif target == "mixer":
            repl_block = AttnBlockWithOutput(block)
            mixer_block = MixerBlock(model_name)
            repl_block.attn = mixer_block
        elif target == "lstm":
            repl_block = AttnBlockWithOutput(block)
            lstm_block = LstmBlock(model_name)
            repl_block.attn = lstm_block
        else:
            raise NotImplementedError("Not available replace architecture (attn/mixer/lstm)")  

        repl_block.to(DEVICE)
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
