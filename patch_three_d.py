# three d modifications to 2D timm specific code

from typing import Callable, Optional

from torch import nn as nn
import torch.nn.functional as F

from timm.layers.helpers import to_3tuple
from timm.layers.trace_utils import _assert


from enum import Enum
from typing import Union

import torch


class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'

    NCDHW = 'NCDHW'
    NDHWC = 'NDHWC'


FormatT = Union[str, Format]


def ncdhw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NDHWC:
        x = x.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC
    return x


class PatchEmbedThreeD(nn.Module):
    """
    3D Patch Embedding Module.

    This module divides a 3D volume into non-overlapping patches, projects each patch into a
    specified embedding dimension, and optionally normalizes the output. It supports both fixed
    image sizes (with strict shape checking) and dynamic padding (if the input dimensions are not
    exactly divisible by the patch size).

    Attributes:
        img_size (tuple[int, int, int] or None): Expected input volume dimensions (Depth, Height, Width).
        patch_size (tuple[int, int, int]): Size of the patch along each dimension.
        grid_size (tuple[int, int, int] or None): Number of patches along each spatial dimension.
        num_patches (int or None): Total number of patches (product of grid_size) if img_size is specified.
        flatten (bool): If True, flattens the spatial dimensions into a sequence.
        output_fmt (Format): Desired output format (e.g., NCDHW or NDHWC).
        strict_img_size (bool): If True, the input dimensions must exactly match img_size.
        dynamic_img_pad (bool): If True, the input volume will be padded to be divisible by patch_size.
        proj (nn.Conv3d): 3D convolution layer that performs the patch projection.
        norm (nn.Module): Normalization layer applied after patch projection.
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 1,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        """
        Initializes the 3D Patch Embedding module.

        Args:
            img_size (Optional[int]): The spatial size of the input volume. If provided, it is converted
                                        to a 3-tuple (D, H, W). If None, grid_size and num_patches will be None.
            patch_size (int): The size of each patch. Converted to a 3-tuple.
            in_chans (int): Number of input channels.
            embed_dim (int): Dimension of the patch embeddings.
            norm_layer (Optional[Callable]): Normalization layer constructor; if None, no normalization is applied.
            flatten (bool): Whether to flatten the output patches into a sequence.
            output_fmt (Optional[str]): Desired output format (e.g., "NCDHW" or "NDHWC"). If provided, flattening
                                        is disabled.
            bias (bool): If True, adds a learnable bias to the convolution projection.
            strict_img_size (bool): If True, asserts that the input dimensions exactly match img_size.
            dynamic_img_pad (bool): If True, pads the input so that its dimensions are divisible by patch_size.
        """
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        if img_size is not None:
            self.img_size = to_3tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            #self.output_fmt = Format.NCHW
            self.output_fmt = Format.NCDHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        Forward pass for the 3D Patch Embedding module.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W].

        Returns:
            torch.Tensor: The embedded patches. If flatten is True, the output is of shape
                          [B, N, embed_dim], where N is the total number of patches.
                          Otherwise, the output format depends on the specified output_fmt.
        """
        B, C, D, H, W = x.shape  # Adjust for depth dimension
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(D == self.img_size[0], f"Input depth ({D}) doesn't match model ({self.img_size[0]}).")
                _assert(H == self.img_size[1], f"Input height ({H}) doesn't match model ({self.img_size[1]}).")
                _assert(W == self.img_size[2], f"Input width ({W}) doesn't match model ({self.img_size[2]}).")
            elif not self.dynamic_img_pad:
                _assert(
                    D % self.patch_size[0] == 0,
                    f"Input depth ({D}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    H % self.patch_size[1] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[1]})."
                )
                _assert(
                    W % self.patch_size[2] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[2]})."
                )
        if self.dynamic_img_pad:
            pad_d = (self.patch_size[0] - D % self.patch_size[0]) % self.patch_size[0]
            pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
            pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        x = self.proj(x)

        if self.flatten:
            # Adjust flattening for 3D data
            x = x.flatten(2).transpose(1, 2)  # NDHWC -> NLC (if you are flattening all spatial dimensions)
        elif self.output_fmt != Format.NDHWC:
            x = ncdhw_to(x, self.output_fmt)  # Assuming ncdhw_to is a function you've defined for 3D data format conversion

        x = self.norm(x)
        return x
