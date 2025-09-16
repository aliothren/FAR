# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2022 paper
# A-ViT: Adaptive Tokens for Efficient Vision Transformer
# Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz,
# and Pavlo Molchanov
# --------------------------------------------------------

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from modeling.avit_vision_transformer import ActVisionTransformer


__all__ = [
    'avit_tiny_patch16_224', \
    'avit_small_patch16_224', \
    'avit_base_patch16_224', \
]


@register_model
def avit_tiny_patch16_224(pretrained=False, **kwargs):

    model = ActVisionTransformer(
        patch_size=16, 
        embed_dim=192, 
        depth=12, 
        num_heads=3,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        # note that this part loads DEIT weights, not A-ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def avit_small_patch16_224(pretrained=False, **kwargs):

    model = ActVisionTransformer(
        patch_size=16, 
        embed_dim=384, 
        depth=12, 
        num_heads=6, 
        **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        # note that this part loads DEIT weights, not A-ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def avit_base_patch16_224(pretrained=False, **kwargs):

    model = ActVisionTransformer(
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        # note that this part loads DEIT weights, not A-ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model
