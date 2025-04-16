# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial

import torch
import torch.nn as nn

from models.utils.patch_embed import PatchEmbed, PatchEmbed3D
from models.utils.modules import Block
from models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from utilities.tensors import trunc_normal_

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size = 182,
        patch_size = 14,
        num_frames = 224,
        tubelet_size=16,
        in_chans = 1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        non_zero_patch_opt=True,
        **kwargs
    ):
        super().__init__()
        self.non_zero_patch_opt = non_zero_patch_opt
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers

        self.input_size = img_size
        self.patch_size = patch_size

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

        grid_size = self.input_size // self.patch_size # this is 13 (182/14)
        grid_depth = self.num_frames // self.tubelet_size # this is 14 (224/16)

        # Tokenize pixels with convolution
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        # self.num_patches is 13 * 14 * 13
        self.num_patches = grid_size * grid_depth * grid_size
        # Position embedding
        self.pos_embed = None
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim),
            requires_grad=False)

        # Attention Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.GELU,
                grid_size=grid_size,
                grid_depth=grid_depth,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # ------ initialize weights
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size
        sincos = get_3d_sincos_pos_embed(
                embed_dim,
                grid_size,
                grid_depth,
                cls_token=False,
            )
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def non_zero_patch(self, x, y):
        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size
        # we want to use only non-zero patches to optimize 
        # non_zero_mask = y.view((-1, 13, 14, 14, 16, 13, 14)).sum((2, 4, 6)) != 0 # all non-zero patches over 3 dimensions
        # all non-zero patches over 3 dimensions
        non_zero_mask = y.view(\
        (-1, grid_size, self.patch_size, grid_depth, self.tubelet_size,\
         grid_size, self.patch_size)).sum((2, 4, 6)) != 0
        #non_zero_mask = non_zero_mask.view((-1, 13*14*13)) # flattening all non-zero patches
        # flattening all non-zero patches
        non_zero_mask = non_zero_mask.view((-1, grid_size * grid_depth * grid_size)) 
        
        #non_zero_patch = torch.where(non_zero_mask == True) 
        # store indices of non-zero patches X,Y coordinates
        
        # if at least one of the patches across images from same batch is non-zero, keep that patch
        batch_mask = non_zero_mask.max(0)[0] # union of non-zero patches over images from same batch
        # Basically, non_zero_mask.max(0)[0] gives you the maximum values along the first 
        # dimension of the non_zero_mask tensor. Recall first dimension of this tensor is which 
        # part of the batch. Eg. 0, 1, 2, etc. depending on the batch size.
        # If non_zero_mask is a tensor of boolean values, 
        # this will effectively return True for a column if there's at least one True value 
        # in that column, and False otherwise.
        x = x[:, batch_mask, :] # selecting non-zero patches
        return x, batch_mask, non_zero_mask

    def forward(self, x, y):
        """
        :param x: input image/video
        :param masks: indices of patch tokens to mask (remove)
        """

        # Tokenize input
        pos_embed = self.pos_embed
        x = self.patch_embed(x)
        if pos_embed is not None:
            x += pos_embed
        B, N, D = x.shape
        if self.non_zero_patch_opt:
            x, batch_mask, non_zero_patch = self.non_zero_patch(x, y)
            
        # Fwd prop
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))

        if self.out_layers is not None:
            return outs

        if self.norm is not None:
            x = self.norm(x)

        # return non_zero_patch: 
        # this way, we keep track of which patches are non-zero across samples in a batch and only those are being used in encoder
        return x, batch_mask, non_zero_patch 

def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_gigantic(patch_size=14, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1664, depth=48, num_heads=16, mpl_ratio=64/13,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
    'vit_gigantic': 1664,
}
