# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2022 paper
# A-ViT: Adaptive Tokens for Efficient Vision Transformer
# Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz,
# and Pavlo Molchanov
# --------------------------------------------------------

# The following snippets are started from:
# https://github.com/facebookresearch/deit
# &
# https://github.com/rwightman/pytorch-image-models
# Before code is extensively modified to accomodate A-ViT training

from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Mlp
from timm.models.vision_transformer import VisionTransformer, Block, Attention

from modeling.utils import get_distribution_target


class Masked_Attention(Attention):
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

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
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

                # attn_mask = mask.view(B, 1, 1, N).expand(B, self.num_heads, N, N)
                # attn_mask = attn_mask.to(torch.bool)
                # x = F.scaled_dot_product_attention(
                #     q, k, v,
                #     attn_mask=attn_mask,
                #     dropout_p=self.attn_drop.p if self.training else 0.,
                # )
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
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block_ACT(Block):
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
        self.attn = Masked_Attention(
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
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        else:
            x = x + self.drop_path1(
                self.ls1(
                    self.attn(
                        self.norm1(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1),
                        mask=mask,
                    ),
                )
            )
            x = x + self.drop_path2(
                self.ls2(
                    self.mlp(
                        self.norm2(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1)
                    )
                )
            )

        if self.act_mode == 4:
            gate_scale, gate_center = self.args.gate_scale, self.args.gate_center
            halting_score_token = self.sig(x[:,:,0] * gate_scale - gate_center)
            halting_score = [-1, halting_score_token]
        else:
            raise NotImplementedError(f"Unsupported act_mode {self.act_mode}")

        return x, halting_score


# Adaptive Vision Transformer
class ActVisionTransformer(VisionTransformer):
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
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block_ACT(
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

    def forward_features(self, x):
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

            # block out all the parts that are not used
            out = out * mask_token.float().view(bs, self.total_token_cnt, 1)

            # evaluate layer and get halting probability for each sample
            # block_output, h_lst = l.forward_act(out)    # h is a vector of length bs, block_output a 3D tensor
            block_output, h_lst = l.forward_act(out, 1.-mask_token.float())    # h is a vector of length bs, block_output a 3D tensor

            if self.args.distr_prior_alpha>0.:
                self.halting_score_layer.append(torch.mean(h_lst[1][1:]))

            out = block_output.clone()              # Deep copy needed for the next layer

            _, h_token = h_lst # h is layer_halting score, h_token is token halting score, first position discarded

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
            delta1 = block_output * R_token.view(bs, self.total_token_cnt, 1) * reached_token.view(bs, self.total_token_cnt, 1)
            self.rho_token = self.rho_token + R_token * reached_token

            # Case 2: threshold not reached
            # token part
            not_reached_token = c_token < 1 - self.eps
            not_reached_token = not_reached_token.float()
            R_token = R_token - (not_reached_token.float() * h_token)
            delta2 = block_output * h_token.view(bs, self.total_token_cnt, 1) * not_reached_token.view(bs, self.total_token_cnt, 1)

            self.counter_token = self.counter_token + not_reached_token # These data points will need at least one more layer

            # Update the mask
            mask_token = (c_token < 1 - self.eps).float()
            mask_token[:, 0] = 1.0
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.act_mode == 4, f'Unsupported act_mode {self.act_mode}'
        x = self.forward_features(x)
        return self.forward_head(x)
