import os
import torch
import config
import architectures
import matplotlib.pyplot as plt

from pathlib import Path
from data import load_dataset
from timm.models import create_model
  

def plot_heatmap(
    data: torch.Tensor, 
    title: str, 
    save_path=None,
    save_cls = True,
    save_patch = True
) -> None:
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    seq_length = data.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='viridis', vmin=0, vmax=2.0)
    # im = ax.imshow(np.log1p(data), cmap='viridis', vmin=0, vmax=2.0)
    ax.set_xticks([0, seq_length - 1])
    ax.set_xticklabels(['0', f'{seq_length - 1}'])
    ax.set_yticks([0, seq_length - 1])
    ax.set_yticklabels(['0', f'{seq_length - 1}'])
    fig.colorbar(im, ax=ax, shrink=0.9)

    plt.title(title)
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to {save_path}")
    plt.close(fig) 
    
    if save_cls and save_path:
        cls_data = data[0]
        cls_path = save_path.with_name(save_path.stem + "_cls" + save_path.suffix)

        fig, ax = plt.subplots()
        ax.plot(cls_data)
        ax.set_title(title + " (CLS Row)")
        ax.set_xlabel("Key Token Index")
        ax.set_ylabel("Attention Weight")
        ax.set_xlim(0, len(cls_data) - 1)
        ax.set_ylim(0, 1)
        ax.grid(True)

        plt.savefig(cls_path, dpi=300, bbox_inches='tight')
        print(f"CLS Attention saved to {cls_path}")
        plt.close(fig)
    
    if save_patch and save_path:
        patch_data = data[0, 1:].reshape(14, 14)  # drop CLS token
        patch_path = save_path.with_name(save_path.stem + "_cls_patch" + save_path.suffix)

        fig, ax = plt.subplots()
        im = ax.imshow(patch_data, cmap='viridis', vmin=0, vmax=0.5)
        # im = ax.imshow(patch_data, cmap='viridis')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.9)

        plt.savefig(patch_path, dpi=300, bbox_inches='tight')
        print(f"Patch Attention saved to {patch_path}")
        plt.close(fig)
        

def plot_attention_heatmap(attentions, head_ids, layer_ids, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for layer_id in layer_ids:
        attention = attentions[layer_id]
        for head_id in head_ids:
            data = attention[head_id].numpy()
            save_path = save_dir / f"layer{layer_id}_head{head_id}.png"
            plot_heatmap(data, f'FAR: Layer {layer_id} Head {head_id}', save_path)
        avg_data = torch.mean(attention, dim=0).numpy()
        save_path = save_dir / f"layer{layer_id}_avg.png"
        plot_heatmap(avg_data, f'Average Gradiant Map: Layer {layer_id}', save_path)


def plot_gradiant_heatmap(gradiants, layer_ids, save_dir, mode):
    os.makedirs(save_dir, exist_ok=True)
    for layer_id in layer_ids:
        gradiant = gradiants[layer_id]
        data = gradiant.numpy()
        save_path = save_dir / f"layer{layer_id}_{mode}.png"
        plot_heatmap(data, f'Average Attention Map: Layer {layer_id}', save_path)
  

def get_gradients_multihead(model, imgs, head_num=3, mode="avg"):
    """
    return grad_maps:  shape = [head_num, num_blocks, T, T]
    """
    model.train() 

    inputs  = [None] * len(model.blocks)
    outputs = [[None]*head_num for _ in model.blocks]
    hooks, original_forwards = [], []
    
    print("Registering hooks and fwds")
    for idx, blk in enumerate(model.blocks):

        def hook_fn(module, input, output, blk_idx=idx):
            input = input[0]
            input.retain_grad()
            inputs[blk_idx] = input
        hooks.append( blk.attn.lstm.register_forward_hook(hook_fn) )

        # ---- monkey-patch block.attn.forward ----
        original_forward = blk.attn.forward
        original_forwards.append(original_forward)

        def new_forward(self, input, blk_idx=idx):
            # lstm_out: [B, T, 2*hidden_dim_total]  (=2*head_num*H)
            lstm_out, _ = self.lstm(self.token_norm(self.pre_proj(input)))
            # H = self.hidden_dim // self.head_num
            H = self.hidden_dim // 3
            fw = lstm_out[:, :, :self.hidden_dim]         # [B, T, hidden_dim]
            bw = lstm_out[:, :, self.hidden_dim:]         # [B, T, hidden_dim]

            for head_idx in range(head_num):
                start = head_idx * H
                end = (head_idx + 1) * H
                fw_part = fw[:, :, start:end]  # [B, T, H]
                bw_part = bw[:, :, start:end]  # [B, T, H]
                # bw_part = torch.flip(bw[:, :, start:end], dims=[1])  # [B, T, H]
                if mode == "forward":
                    outputs[blk_idx][head_idx] = fw_part
                elif mode == "backward":
                    outputs[blk_idx][head_idx] = bw_part
                elif mode == "avg":
                    outputs[blk_idx][head_idx] = torch.cat([fw_part, bw_part], dim=-1)
                     
            return self.proj(lstm_out) 

        blk.attn.forward = new_forward.__get__(blk.attn, blk.attn.__class__)

    print("Calculating gradiants")
    gradiants = []
    for b in range(imgs.shape[0]): # batchsize
        img = imgs[b].unsqueeze(0).detach().clone().requires_grad_(True)
        inputs  = [None] * len(model.blocks)
        outputs = [[None] * head_num for _ in model.blocks] 
        with torch.enable_grad():
            _ = model(img)

        gradiant = [ [] for _ in range(head_num) ]
        for blk_idx in range(len(model.blocks)):
            blk_input = inputs[blk_idx]
            blk_output = outputs[blk_idx]
            
            for head_idx in range(head_num):
                print(f"Calculating grad of img {b} for block {blk_idx}, head {head_idx}")
                head_output = blk_output[head_idx] # [1,T,*]
                token_num = head_output.shape[1]
                layer_grad = torch.zeros((token_num, token_num))
                
                for token_idx in range(token_num):
                    model.zero_grad()
                    if blk_input.grad is not None: 
                        blk_input.grad.zero_()
                    token = head_output[0, token_idx]
                    token.sum().backward(retain_graph=True)
                    token_grad = blk_input.grad.detach().abs().sum(dim=-1).squeeze(0) 
                    layer_grad[token_idx] = token_grad.cpu() 
                        
                gradiant[head_idx].append(layer_grad)

        gradiants.append(torch.stack([torch.stack(h) for h in gradiant]))
    gradiants = torch.stack(gradiants, dim=0) 
    avg_gradiant = gradiants.mean(dim=0).permute(1, 0, 2, 3).contiguous()
    
    for h in hooks:
        h.remove()
    for i, blk in enumerate(model.blocks):
        blk.attn.forward = original_forwards[i]

    model.eval()
    return avg_gradiant   


def get_attentions(model, imgs):
    hooks = []
    attn_scores = []
    model.eval()
    # monkey patch Attention.forward
    def hook_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_scores.append(attn.detach().cpu()) 
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    backup = []
    for blk in model.blocks:
        backup.append(blk.attn.forward)
        blk.attn.forward = hook_forward.__get__(blk.attn, blk.attn.__class__) 

    with torch.no_grad():
        _ = model(imgs)

    for h in hooks:
        h.remove()
        
    for i in range(len(attn_scores)):
        attn = attn_scores[i]  # shape: [B, H, N, N]
        if attn.shape[0] > 1:  
            attn_scores[i] = attn.mean(dim=0)  # → [H, N, N]
        else: 
            attn_scores[i] = attn[0]  # → [H, N, N]
    return attn_scores                 


if __name__ == "__main__":
    # Modify visualization scope here
    layers = list(range(12))
    heads = [0, 1, 2]
    batch = 1

    parser = config.get_args_parser()
    args = parser.parse_args()
    args = config.fill_default_args(args)
    
    # Load data
    data_loader_val, _ = load_dataset(args, "val")
    imgs, targets = next(iter(data_loader_val)) 
    imgs = imgs[0:batch]
    imgs = imgs.to(args.device)
    
    # Load model
    print(f"Creating model: {args.vis_model}")
    print(f"Using weight: {args.vis_weight}")
    if "DeiT" in args.vis_model:
        model = create_model(
            model_name=args.vis_model_name, pretrained=False, num_classes=args.nb_classes, drop_rate=args.drop,
            drop_path_rate=args.drop_path, drop_block_rate=None, img_size=args.input_size
            )
        model = architectures.load_weight(model, args.vis_weight)
    else:
        model = torch.load(args.vis_weight)
    model.to(args.device)

    output_dir = Path(args.base_dir) / "figs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = output_dir / f"{args.vis_model}"
    
    # Get attention scores
    if "DeiT" in args.vis_model:
        save_path = save_path / "uni"
        attentions =  get_attentions(model, imgs) 
        # Save heatmap
        plot_attention_heatmap(attentions, heads, layers, save_path)
    elif "Multihead" in args.vis_model:
        mode = "avg"
        save_path = save_path / "uni"
        gradiants = get_gradients_multihead(model, imgs, mode=mode) 
        plot_attention_heatmap(gradiants, heads, layers, save_path)
        