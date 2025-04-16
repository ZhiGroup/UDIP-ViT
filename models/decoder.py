import numpy as np
import torch
import torch.nn as nn
from models.utils.modules import Block
from models.utils.pos_embs import get_3d_sincos_pos_embed  # Import the function from pos_emb.py
from utilities.tensors import trunc_normal_
import math

class Decoder(nn.Module):
    def __init__(self, 
                 img_size = 182,
                 patch_size = 14,
                 num_frames = 224,
                 tubelet_size=16,
                 embed_dim=768,
                 decoder_embed_dim=384,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 init_std=0.02,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        # Initialize pos_emb if pos is provided
        self.input_size = img_size
        self.patch_size = patch_size

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.num_patches = (self.input_size // self.patch_size) ** 2 * (self.num_frames // self.tubelet_size)
        grid_size = self.input_size // self.patch_size # this is 13 (182/14)
        grid_depth = self.num_frames // self.tubelet_size # this is 14 (224/16)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim),
            requires_grad=False)
        self._init_pos_embed(self.pos_emb.data)
        self.blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
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
        self.norm = norm_layer(decoder_embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        patch_numel = self.patch_size ** 2 * self.tubelet_size
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_numel, bias=True)
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

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

    def forward(self, x, batch_mask):
        pos_emb = self.pos_emb[:, batch_mask, :].repeat(x.shape[0], 1, 1)
        num_latent = x.shape[1]
        x = self.decoder_embed(x)
        x = torch.cat([x, pos_emb], dim=1)
        for blk in self.blocks:
            x = blk(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.decoder_pred(x[:, num_latent:, :])
        return x
