# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            #kernel_size=(tubelet_size, patch_size, patch_size),
            kernel_size=(patch_size, tubelet_size, patch_size),
            #stride=(tubelet_size, patch_size, patch_size),
            stride=(patch_size, tubelet_size, patch_size),
        )

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1) # add this dimension because x shape is expected to be 5 dimensions
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        # NOTE: x shape after proj(x) is:
        # (in_chans, emdeb_dim, patch_size, tubelet_size, patch_size).
        # The flatten(2) step flattens along dimension 2. So at this
        # stage, shape of proj(x).flatten(2) is:
        # (in_chans, embed_sim, patch_size * tubelet_size * patch_size).
        # After transpose(1,2) step, dimensions 1 and 2 are transposed.
        # So shape of proj(x).flatten(2).transpose(1,2) is:
        # (in_chans, patch_size * tubelet_size * patch_size, embed_dim).

        return x
