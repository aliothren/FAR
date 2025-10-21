import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import collections

from timm.layers import Mlp
from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer, Block, Attention

from typing import Optional, Type
from mamba_ssm import Mamba2

from modeling.utils import get_distribution_target


# Bidirectional Mamba module using Mamba-2 block, same architecture as BiLSTM
class BiMamba(nn.Module):
    """
    Drop-in replacement for DeiT attention, using official Mamba-2 block.
      - Use bidirectional Mamba (like BiLSTM), and post projection after mamba to keep same shape
    """
    def __init__(
        self,
        original_attn: nn.Module,
        d_conv: int = 4,
        d_state: int = 256,
        expand: int = 1,
        n_groups: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = original_attn.qkv.in_features
        self.head_num = original_attn.num_heads
        self.head_dim = self.input_dim // self.head_num
        self.hidden_dim = d_state if d_state is not None else self.head_dim
        self.output_dim = self.input_dim
        self.n_groups = 1 if not n_groups else self.head_num

        self.mamba_fwd = Mamba2(
            d_model=self.input_dim, 
            d_state=self.hidden_dim, 
            d_conv=d_conv, 
            expand=expand,
            headdim=self.head_dim,
            d_ssm=self.input_dim,
            ngroups=self.n_groups,
            use_mem_eff_path=False,
        )
        self.mamba_fwd.out_proj = nn.Identity()
        self.mamba_bwd = Mamba2(
            d_model=self.input_dim, 
            d_state=self.hidden_dim, 
            d_conv=d_conv, 
            expand=expand,
            headdim=self.head_dim,
            d_ssm=self.input_dim,
            ngroups=self.n_groups,
            use_mem_eff_path=False,
        )
        self.mamba_bwd.out_proj = nn.Identity()
        self.head_proj = nn.Conv1d(
                            in_channels=2 * self.input_dim, 
                            out_channels=self.input_dim, 
                            kernel_size=1, 
                            groups=self.head_num, 
                            bias=False,
                        )
        self.post_proj = original_attn.proj

    def forward(self, x: torch.Tensor):
        x_fwd, _ = self.mamba_fwd(x)  # [B, N, C]
        x_bwd, _ = self.mamba_bwd(torch.flip(x, dims=[1]))  # [B, N, C]
        x = torch.cat([x_fwd, torch.flip(x_bwd, dims=[1])], dim=-1)  # [B, N, 2C]
        x = self.head_proj(x.transpose(1, 2)).transpose(1, 2)    
        self.attn_out = x.clone()
        out = self.post_proj(x)

        return out, self.attn_out


# Mamba-2 block with intermediate output
class MambaWithOutput(nn.Module):
    """
    Drop-in replacement for DeiT attention, using official Mamba-2 block.
      - Implement inter-layer flipping (odd/even) without changing shapes
    """

    def __init__(
        self,
        original_attn: nn.Module,
        layer_id: int = 0,
        d_conv: int = 4,
        d_state: int = 256,
        expand: int = 1,
        n_groups: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = original_attn.qkv.in_features
        self.head_num = original_attn.num_heads
        self.head_dim = self.input_dim // self.head_num
        self.hidden_dim = d_state if d_state is not None else self.head_dim
        self.output_dim = self.input_dim
        self.layer_id = layer_id
        self.n_groups = 1 if not n_groups else self.head_num

        self.mamba = Mamba2(
            d_model=self.input_dim, 
            d_state=self.hidden_dim, 
            d_conv=d_conv, 
            expand=expand,
            headdim=self.head_dim,
            d_ssm=self.input_dim,
            ngroups=self.n_groups,
            use_mem_eff_path=False,
        )

    def forward(self, x: torch.Tensor):
        # Inter-layer flipping
        if self.layer_id % 2 == 1:
            x = torch.flip(x, dims=[1])
        out, self.attn_out = self.mamba(x)  # [B, N, C]

        return out, self.attn_out


# Multi-Head LSTM with intermediate output
class MultiHeadLstm(nn.Module):
    def __init__(self, original_attn, num_layers=1, dropout=0.1):
        super(MultiHeadLstm, self).__init__()
        self.input_dim = original_attn.qkv.in_features
        self.head_num = original_attn.num_heads
        self.hidden_dim = original_attn.head_dim * self.head_num
        self.output_dim = self.input_dim

        mask_ih = get_block_mask(
            self.input_dim // self.head_num,
            self.hidden_dim // self.head_num,
            self.head_num,
        )
        mask_hh = get_block_mask(
            self.hidden_dim // self.head_num,
            self.hidden_dim // self.head_num,
            self.head_num,
        )
        mask_head = get_head_mask(self.hidden_dim // self.head_num, self.head_num)

        self.pre_proj = nn.Linear(self.input_dim, self.input_dim)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head_proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.post_proj = original_attn.proj

        for name, param in self.named_parameters():
            if "weight_ih" in name:
                param.data *= mask_ih
            elif "weight_hh" in name:
                param.data *= mask_hh
            elif "head_proj.weight" in name:
                param.data *= mask_head

    def forward(self, x):
        x = self.pre_proj(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.head_proj(lstm_out)
        self.lstm_out = lstm_out.clone()
        out = self.post_proj(lstm_out)
        return out, self.lstm_out


# Parallel Multi-Head LSTM for hardware deployment
class ParallelLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        pre_proj: nn.Linear,
        head_proj: nn.Linear,
        post_proj: nn.Linear,
        dropout=0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=input_dim // num_heads,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.head_projs = nn.ModuleList(
            [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_heads)]
        )
        self.post_proj = post_proj
        self.pre_proj = (
            pre_proj if pre_proj is not None else nn.Linear(input_dim, input_dim)
        )

    def forward(self, x):
        x = self.pre_proj(x)
        self.pre_proj_out = x.clone()
        chunks = torch.chunk(x, self.num_heads, dim=-1)

        outs = []
        for lstm, head_proj, chunk in zip(self.lstms, self.head_projs, chunks):
            y = lstm(chunk)[0]
            y = head_proj(y)
            outs.append(y)
        y_cat = torch.cat(outs, dim=-1)  # [B, N, 192]

        return self.post_proj(y_cat)


# DeiT attention part with intermediate output
class AttentionWithOutput(nn.Module):
    "DeiT attention part with intermediate output"

    def __init__(self, original_attn) -> None:
        super().__init__()
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.scale = original_attn.scale

        self.qkv = original_attn.qkv
        self.proj = original_attn.proj
        self.attn_drop = original_attn.attn_drop
        self.proj_drop = original_attn.proj_drop

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        self.attn_out = x.clone()
        out = self.proj(x)
        out = self.proj_drop(out)
        return out, self.attn_out


# DeiT Block with intermediate output
class BlockWithOutput(nn.Module):
    """DeiT Block with shortcut"""

    def __init__(self, original_block, target, layer_id=0):
        super(BlockWithOutput, self).__init__()
        if target == "attn":
            self.attn = AttentionWithOutput(original_block.attn)
        elif target == "multi-lstm":
            self.attn = MultiHeadLstm(original_block.attn)
        elif target == "mamba":
            # self.attn = MambaWithOutput(original_block.attn, layer_id=layer_id)
            self.attn = BiMamba(original_block.attn)
        else:
            raise NotImplementedError(
                "Not available replace architecture (attn/multi-lstm)"
            )

        self.mlp = original_block.mlp
        self.norm1 = original_block.norm1
        self.norm2 = original_block.norm2
        self.drop_path = getattr(original_block, "drop_path", nn.Identity())
        self.block_output = None

    def forward(self, x):
        y = self.norm1(x)
        y, self.attn_output = self.attn(y)
        x = x + self.drop_path(y)
        y = self.norm2(x)
        x = x + self.drop_path(self.mlp(y))
        self.block_output = x.clone()
        return x


# Adaptive Attention part for A-ViT
class ACT_Attention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mask=None, 
        masked_softmax_bias: float=-1000.,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.mask = mask
        self.masked_softmax_bias = masked_softmax_bias

    def forward(self, x: torch.Tensor, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            if mask is not None:
                add_mask = mask.view(B, 1, 1, N).expand(B, self.num_heads, N, N)  # [B,H,N,N]
                add_mask = add_mask.float()
                add_mask = add_mask.masked_fill(add_mask > 0, float('-inf')).masked_fill(add_mask == 0, 0.0)  # 0 / -inf

                xf = F.scaled_dot_product_attention(
                    q.float(), k.float(), v.float(),
                    attn_mask=add_mask,  # additive mask, same dtype as q/k/v (这里用 fp32)
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
                x = xf.to(q.dtype)  # 回到原精度

            else:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn + mask.view(mask.shape[0], 1, 1, mask.shape[1]) * self.masked_softmax_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        self.attn_output = x.clone()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, self.attn_output


# Adaptive Multi-Head LSTM for A-ViT
class ACT_MultiHeadLstm(nn.Module):
    """
    Multi-Head LSTM with A-ViT style Token-Act
    """
    def __init__(self, original_attn, num_layers: int = 1, dropout: float = 0.0):
        super(ACT_MultiHeadLstm, self).__init__()
        self.input_dim = original_attn.qkv.in_features
        self.head_num = original_attn.num_heads
        self.hidden_dim = original_attn.head_dim * self.head_num
        self.output_dim = self.input_dim

        mask_ih = get_block_mask(
            self.input_dim // self.head_num,
            self.hidden_dim // self.head_num,
            self.head_num,
        )
        mask_hh = get_block_mask(
            self.hidden_dim // self.head_num,
            self.hidden_dim // self.head_num,
            self.head_num,
        )
        mask_head = get_head_mask(self.hidden_dim // self.head_num, self.head_num)
        
        self.pre_proj = nn.Linear(self.input_dim, self.input_dim)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head_proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.post_proj = original_attn.proj

        for name, param in self.named_parameters():
            if "weight_ih" in name:
                param.data *= mask_ih
            elif "weight_hh" in name:
                param.data *= mask_hh
            elif "head_proj.weight" in name:
                param.data *= mask_head

    @torch.no_grad()
    def _build_batch_indices(self, keep_mask: torch.Tensor):
        B, N = keep_mask.shape
        keep_mask = keep_mask.to(dtype=torch.int64)
        if (keep_mask[:, 0] == 0).any():
            keep_mask[:, 0] = 1

        lengths = keep_mask.sum(dim=1).clamp_min(1) 
        L_max = int(lengths.max().item())

        pos = torch.arange(N, device=keep_mask.device).unsqueeze(0).expand(B, -1) 
        big = N * 2
        order_key = pos + (1 - keep_mask) * big  
        if hasattr(torch, 'stable'):
            sorted_idx = order_key.argsort(dim=1, stable=True)
        else:
            sorted_idx = order_key.argsort(dim=1)
        idx_pad = sorted_idx[:, :L_max]
        return idx_pad, lengths, L_max

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, N, C = x.shape

        x = self.pre_proj(x)  # [B, N, C]
        if mask is None:
            keep_mask = x.new_ones(B, N, dtype=torch.int64)
        else:
            keep_mask = (mask < 0.5).to(dtype=torch.int64)

        idx_pad, lengths, L_max = self._build_batch_indices(keep_mask)    # idx_pad:[B,L_max]

        idx_exp = idx_pad.unsqueeze(-1).expand(-1, -1, C)                 # [B, L_max, C]
        x_kept = x.gather(dim=1, index=idx_exp)                           # [B, L_max, C]

        packed = pack_padded_sequence(
                    x_kept, 
                    lengths.to('cpu'),
                    batch_first=True, enforce_sorted=False
                )
        packed_out, _ = self.lstm(packed)
        out_padded, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=L_max) 

        out_kept = self.head_proj(out_padded)
        full_feat = x.new_zeros(B, N, self.hidden_dim)
        full_feat.scatter_(dim=1, index=idx_exp, src=out_kept)

        self.lstm_out = full_feat.clone() # [B, N, C]
        out = self.post_proj(full_feat)

        return out, self.lstm_out
    

# Adaptive Block for A-ViT
class ACT_Block(Block):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            args=None, 
            index=-1,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )
        self.attn = ACT_Attention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            )
        self.act_mode = args.act_mode
        assert self.act_mode in {1, 2, 3, 4} 
        self.index=index
        self.args = args
        if self.act_mode == 4:
            self.sig = torch.sigmoid
        else:
            raise NotImplementedError(f"Unsupported act_mode {self.act_mode}")

    def forward_act(self, x: torch.Tensor, mask=None):

        bs, token, dim = x.shape

        if mask is None:
            y = self.norm1(x)
            y, self.attn_output = self.attn(y)
            x = x + self.drop_path1(self.ls1(y))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        else:
            keep = (1-mask).view(bs, token, 1)
            y = self.norm1(x * keep) * keep
            y, self.attn_output = self.attn(y, mask=mask)
            x = x + self.drop_path1(self.ls1(y))
            x = x + self.drop_path2(
                self.ls2(
                    self.mlp(
                        self.norm2(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1)
                    )
                )
            )
        self.block_output = x.clone()

        if self.act_mode == 4:
            gate_scale, gate_center = self.args.gate_scale, self.args.gate_center
            halting_score_token = self.sig(x[:,:,0] * gate_scale - gate_center)
            halting_score = [-1, halting_score_token]
        else:
            raise NotImplementedError(f"Unsupported act_mode {self.act_mode}")

        return x, halting_score


# Adaptive Vision Transformer for A-ViT
class ACT_VisionTransformer(VisionTransformer):
    """ Vision Transformer with Adaptive Token Capability

    Starting at:
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929

        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877

    Extended to:
        Accomodate adaptive token inference
    """

    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            num_classes=1000, 
            embed_dim=768, 
            depth=12,
            num_heads=12, 
            drop_rate=0., 
            drop_path_rate: float = 0.,
            args=None,
            qkv_bias=True,
            use_external_mask=False, 
            **kwargs,
        ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
        )
        self.args = args
        self.use_external_mask = use_external_mask
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            ACT_Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                proj_drop=drop_rate,
                drop_path=dpr[i],
                args=args, 
                qkv_bias=qkv_bias,
                index=i, 
            )
            for i in range(depth)])

        self.eps = 0.01
        self.rho = None
        self.counter = None  # Keeps track of how many layers are used for each example (for logging)
        self.batch_cnt = 0 # amount of batches seen, mainly for tensorboard

        # for token act part
        self.c_token = None
        self.R_token = None
        self.mask_token = None
        self.rho_token = None
        self.counter_token = None
        self.total_token_cnt = self.patch_embed.num_patches + self.num_prefix_tokens

        if args.distr_prior_alpha >0. :    
            self.register_buffer(
                'distr_target',
                torch.tensor(get_distribution_target(standardized=True), dtype=torch.float32)
            )
            self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        self.num_classes = num_classes
        self.act_mode = args.act_mode

    def forward_features(self, x, external_masks=None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        # now start the act part
        bs = x.size()[0]  # The batch size
        device = x.device

        # this part needs to be modified for higher GPU utilization
        if self.c_token is None or bs != self.c_token.size()[0]:
            self.c_token = torch.zeros(bs, self.total_token_cnt, device=device)
            self.R_token = torch.ones(bs, self.total_token_cnt, device=device)
            self.mask_token = torch.ones(bs, self.total_token_cnt, device=device)
            self.rho_token = torch.zeros(bs, self.total_token_cnt, device=device)
            self.counter_token = torch.ones(bs, self.total_token_cnt, device=device)

        self._last_masks = []
        self._last_h_tokens = []
        self._self_masks = []
        self._self_h_tokens = []
        
        c_token = self.c_token.clone()
        R_token = self.R_token.clone()
        mask_token = self.mask_token.clone()
        self.rho_token = self.rho_token.detach() * 0.
        self.counter_token = self.counter_token.detach() * 0 + 1.
        # Will contain the output of this residual layer (weighted sum of outputs of the residual blocks)
        output = None
        # Use out to backbone
        out = x

        if self.args.distr_prior_alpha>0.:
            self.halting_score_layer = []

        for i, l in enumerate(self.blocks):
            # use external mask if provided
            if self.use_external_mask:
                assert external_masks is not None, "External masks not provided!"    
                if isinstance(external_masks, (list, tuple)):
                    assert len(external_masks) == len(self.blocks), \
                        f"external_masks 层数不匹配: {len(external_masks)} vs {len(self.blocks)}"
                    layer_keep = external_masks[i]
                else:
                    raise NotImplementedError("Only list/tuple external_masks is supported!")
                if layer_keep.dim() == 1:
                    layer_keep = layer_keep[None, :].expand(bs, -1)
                layer_keep = layer_keep.to(device=device, dtype=mask_token.dtype)
                layer_keep[:, 0] = 1.0
                mask_token = layer_keep
            self._last_masks.append(mask_token.detach()) 

            # block out all the parts that are not used
            out = out * mask_token.float().view(bs, self.total_token_cnt, 1)
            block_output, h_lst = l.forward_act(out, 1.-mask_token.float()) 
            if self.args.distr_prior_alpha>0.:
                self.halting_score_layer.append(h_lst[1][:, 1:].mean()) 

            out = block_output.clone()

            _, h_token = h_lst
            self._last_h_tokens.append(h_token.detach())

            # here, 1 is remaining, 0 is blocked
            block_output = block_output * mask_token.float().view(bs, self.total_token_cnt, 1)

            # Is this the last layer in the block?
            if i==len(self.blocks)-1:
                h_token = torch.ones(bs, self.total_token_cnt, device=device)

            # for token part
            c_token = c_token + h_token
            self.rho_token = self.rho_token + mask_token.float()

            # Case 1: threshold reached in this iteration
            # token part
            reached_token = c_token > 1 - self.eps
            reached_token = reached_token.float() * mask_token.float()
            delta1 = block_output \
                    * R_token.view(bs, self.total_token_cnt, 1) \
                    * reached_token.view(bs, self.total_token_cnt, 1)
            self.rho_token = self.rho_token + R_token * reached_token
            self_mask = (c_token < 1 - self.eps).float()
            self_mask[:, 0] = 1.0
            self._self_masks.append(self_mask)
            self._self_h_tokens.append(h_token)

            # Case 2: threshold not reached
            # token part
            not_reached_token = c_token < 1 - self.eps
            not_reached_token = not_reached_token.float()
            R_token = R_token - (not_reached_token.float() * h_token)
            delta2 = block_output \
                    * h_token.view(bs, self.total_token_cnt, 1) \
                    * not_reached_token.view(bs, self.total_token_cnt, 1)

            self.counter_token = self.counter_token + not_reached_token

            # Update the mask
            if not self.use_external_mask:
                mask_token = (c_token < 1 - self.eps).float()
                mask_token[:, 0] = 1.0   # always keep the cls token
            if output is None:
                output = delta1 + delta2
            else:
                output = output + (delta1 + delta2)

        x = self.norm(output)
        return x


    def forward_probs(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        out_lst = []
        for i, l in enumerate(self.blocks):
            # evaluate layer and get halting probability for each sample
            out = l.forward(x)    # h is a vector of length bs, block_output a 3D tensor
            tmp_prob = self.forward_head(self.norm(out))
            out_lst.append(tmp_prob)
            x = out

        return out_lst


    def forward(self, x: torch.Tensor, external_masks=None) -> torch.Tensor:
        assert self.act_mode == 4, f'Unsupported act_mode {self.act_mode}'
        x = self.forward_features(x, external_masks=external_masks)
        return self.forward_head(x)


def load_weight(model, weight):
    if weight.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            weight, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(weight, map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        checkpoint_model = checkpoint.state_dict()
    elif isinstance(checkpoint, (dict, collections.OrderedDict)):
        if "model" in checkpoint:
            m = checkpoint["model"]
            if isinstance(m, nn.Module):
                checkpoint_model = m.state_dict()
            else:
                checkpoint_model = m
        elif "state_dict" in checkpoint:
            checkpoint_model = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint_model = checkpoint["model_state_dict"]
        else:
            checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches**0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(
        0, 3, 1, 2
    )
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model["pos_embed"] = new_pos_embed

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint_model, strict=False
    )
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    return model


# Block mask for multihead LSTM
def get_block_mask(in_per_head, hid_per_head, head_num):
    H_total = hid_per_head * head_num
    mask = torch.zeros(4 * H_total, in_per_head * head_num)  # 768×192
    for h in range(head_num):
        col = slice(h * in_per_head, (h + 1) * in_per_head)
        for g in range(4):  # i f g o
            row = slice(
                g * H_total + h * hid_per_head, g * H_total + (h + 1) * hid_per_head
            )
            mask[row, col] = 1.0
    return mask


# Head mask for multihead LSTM
def get_head_mask(hid_per_head, head_num):
    forward_mask = torch.block_diag(
        *[torch.ones(hid_per_head, hid_per_head) for _ in range(head_num)]
    )
    backward_mask = torch.block_diag(
        *[torch.ones(hid_per_head, hid_per_head) for _ in range(head_num)]
    )
    full_mask = torch.cat([forward_mask, backward_mask], dim=1)
    return full_mask


def replace_attention(args, model, repl_blocks, target=None):
    print(f"Replacing blocks: {repl_blocks}; Replace by: {target}")
    if target == "avit":
        return model  # no need to replace
    for idx, blk_index in enumerate(repl_blocks):
        block = model.blocks[blk_index]
        if idx == 0:
            print(
                f"Replacing setting: input_dim {block.attn.qkv.in_features},\
                    num_heads {block.attn.num_heads}, \
                    head_dim {block.attn.head_dim},\
                    target {target},\
                    using avit {args.avit}"
            )
        if args.avit:
            if target == "multi-lstm":
                block.attn = ACT_MultiHeadLstm(block.attn)
            elif target == "attn":
                return model  # no need to replace
            else:
                raise NotImplementedError(
                    f"Not available replace architecture {target}"
                )
            repl_block = block
        else:
            repl_block = BlockWithOutput(block, target, layer_id=blk_index)
        repl_block.to(args.device)
        model.blocks[blk_index] = repl_block
    return model


def set_requires_grad(
    model, mode="train", target_blocks=[], target_part="attn", trainable=True
):
    if model is None:
        print("Model is None, cannot set trainable parts.")
        return

    raw_model = model.module if hasattr(model, "module") else model

    target_names = [f"blocks.{block}." for block in target_blocks]

    print("Trainable Params:")
    # Global fintune when transfer to downstream datasets
    if mode == "downstream":
        if target_part == "full":
            for name, param in raw_model.named_parameters():
                param.requires_grad = trainable
        elif target_part == "FC":
            for name, param in raw_model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    if "mlp" in name:
                        param.requires_grad = trainable
                        print(name)
            for name, param in raw_model.head.named_parameters():
                param.requires_grad = True
                print(name)
        elif target_part == "FC+head":
            for name, param in raw_model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    if "mlp" in name:
                        param.requires_grad = trainable
                        print(name)
            for name, param in raw_model.head.named_parameters():
                param.requires_grad = True
                print(name)

    elif mode == "prune":
        for name, param in raw_model.named_parameters():
            param.requires_grad = trainable

    elif mode == "finetune":
        # turn the classification head to trainable
        if target_part == "head":
            for param in raw_model.parameters():
                param.requires_grad = False
            for name, param in raw_model.head.named_parameters():
                param.requires_grad = True
                print(name)
        # turn the whole blocks to trainable
        elif target_part == "sequential":
            for name, param in raw_model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    param.requires_grad = trainable
                    print(name)

    elif mode == "train":
        # turn the whole block to trainable
        if target_part == "full":
            for name, param in raw_model.named_parameters():
                param.requires_grad = trainable
                print(name)
        elif target_part == "block":
            for name, param in raw_model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    param.requires_grad = trainable
                    print(name)
        # turn the replaced part to trainable
        elif target_part == "attn":
            for name, param in raw_model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    if (
                        "attn" in name
                        and "post_proj" not in name
                        and "teacher" not in name
                    ):
                        param.requires_grad = trainable
                        print(name)
        # turn the FC layers in replaced block to trainable
        elif target_part == "FC":
            for name, param in raw_model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    if "mlp" in name:
                        param.requires_grad = trainable
                        print(name)
        elif target_part == "FC+head":
            for name, param in raw_model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    if "mlp" in name:
                        param.requires_grad = trainable
                        print(name)
            for name, param in raw_model.head.named_parameters():
                param.requires_grad = True
                print(name)

    else:
        raise NotImplementedError(
            "Not available set_requires_grad mode (train/finetune/downstream)"
        )


def load_downstream_model(model_path, args, source="local", model_name=""):
    if source == "local":
        model = torch.load(model_path)
    elif source == "online":
        model = create_model(
            model_name=model_name,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size,
        )

    if hasattr(model, "module"):
        model = model.module

    embed_dim = model.head.in_features
    out_dim = args.nb_classes
    model.head = nn.Linear(embed_dim, out_dim)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)
    return model
