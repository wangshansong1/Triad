from __future__ import print_function

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

from PlainConvUNet_load import PlainConvEncoder
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op


class UNetEncoderForMAE(nn.Module):
    def __init__(
        self,
        in_chans=1,
        patch_size=(2, 2, 2),
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_kernel_sizes=((3, 3, 3),) * 6,
        pool_op_kernel_sizes=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        n_conv_per_stage=(2, 2, 2, 2, 2, 2),
    ):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.skip_channels = tuple(features_per_stage)
        self.out_channels = features_per_stage[-1]

        self.encoder = PlainConvEncoder(
            input_channels=in_chans,
            n_stages=len(conv_kernel_sizes),
            features_per_stage=list(features_per_stage),
            conv_op=convert_dim_to_conv_op(len(conv_kernel_sizes[0])),
            kernel_sizes=list(conv_kernel_sizes),
            strides=list(pool_op_kernel_sizes),
            n_conv_per_stage=list(n_conv_per_stage),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            return_skips=True,
            nonlin_first=False,
        )

        self.mask_token_vol = nn.Parameter(torch.zeros(1, in_chans, 1, 1, 1))
        nn.init.trunc_normal_(self.mask_token_vol, mean=0.0, std=0.02)
        self.z_proj = nn.Conv3d(features_per_stage[-1], features_per_stage[-1], kernel_size=1, bias=True)

    def _apply_image_mask(self, x, mask):
        voxel_mask = (
            mask.repeat_interleave(self.patch_size[0], 1)
            .repeat_interleave(self.patch_size[1], 2)
            .repeat_interleave(self.patch_size[2], 3)
            .unsqueeze(1)
            .to(x.dtype)
        )
        return x * (1.0 - voxel_mask) + self.mask_token_vol * voxel_mask

    def forward(self, x, mask, choice):
        if choice == "mae":
            assert mask is not None, "mask required in mae choice"
            x = self._apply_image_mask(x, mask)

        skips = self.encoder(x)
        z = self.z_proj(skips[-1])
        return z, skips


class SwinTransformerSkipForSimMIM(nn.Module):
    def __init__(self, embed_dim=None):
        super(SwinTransformerSkipForSimMIM, self).__init__()
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)
        self.out_channels = embed_dim * 16
        self.skip_channels = (
            embed_dim,
            embed_dim * 2,
            embed_dim * 4,
            embed_dim * 8,
            self.out_channels,
        )

        self.swinViT = SwinViT(
            in_chans=1,
            embed_dim=self.embed_dim,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=True,
            spatial_dims=3,
            use_v2=True,
        )

    def forward(self, x, mask, choice):
        _, _, d, h, w = x.size()
        x = self.swinViT.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        batch_size, num_tokens, _ = x.shape
        if choice == "mae":
            assert mask is not None
            mask_tokens = self.mask_token.expand(batch_size, num_tokens, -1)
            voxel_mask = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
            x = x * (1.0 - voxel_mask) + mask_tokens * voxel_mask

        x = rearrange(x, "b (d h w) c -> b c d h w", d=d // 2, h=h // 2, w=w // 2)

        x0 = self.swinViT.pos_drop(x)
        if self.swinViT.use_v2:
            x0 = self.swinViT.layers1c[0](x0.contiguous())
        x1 = self.swinViT.layers1[0](x0.contiguous())
        if self.swinViT.use_v2:
            x1 = self.swinViT.layers2c[0](x1.contiguous())
        x2 = self.swinViT.layers2[0](x1.contiguous())
        if self.swinViT.use_v2:
            x2 = self.swinViT.layers3c[0](x2.contiguous())
        x3 = self.swinViT.layers3[0](x2.contiguous())
        if self.swinViT.use_v2:
            x3 = self.swinViT.layers4c[0](x3.contiguous())
        x4 = self.swinViT.layers4[0](x3.contiguous())
        x4_out = self.swinViT.proj_out(x4, True)

        return x4_out, [x0, x1, x2, x3, x4_out]

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {"mask_token"}


def L2dist(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    batch_size = x1.size(0)
    diff = (x1 - x2).reshape(batch_size, -1)
    out = (diff * diff).sum(dim=1)
    return torch.sqrt(out + eps / batch_size)


def LogRatioLoss(input: torch.Tensor, gt_dist: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    batch_size = input.size(0)

    idx1 = torch.randperm(batch_size, device=input.device)
    idx2 = torch.randperm(batch_size, device=input.device)

    f_i = input[idx1]
    f_j = input[idx2]

    y_i = gt_dist[idx1]
    y_j = gt_dist[idx2]

    Dfai = L2dist(input, f_i)
    Dfaj = L2dist(input, f_j)
    Dyai = L2dist(gt_dist, y_i)
    Dyaj = L2dist(gt_dist, y_j)

    log_dist = torch.log(Dfai.clamp_min(epsilon)) - torch.log(Dfaj.clamp_min(epsilon))
    log_gt_dist = torch.log(Dyai.clamp_min(epsilon)) - torch.log(Dyaj.clamp_min(epsilon))

    return F.mse_loss(log_dist, log_gt_dist)


class SimMIMSkip_light(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder,
        feature_size: int = 24,
        choice="mae",
        ssl_weight: float = 0.0,
        text_dim: int | None = None,
    ):
        super().__init__()
        self.encoder = encoder

        self.in_chans = in_channels
        if hasattr(self.encoder, "patch_size"):
            self.patch_size = self.encoder.patch_size
        elif hasattr(self.encoder, "swinViT") and hasattr(self.encoder.swinViT, "patch_size"):
            self.patch_size = self.encoder.swinViT.patch_size
        else:
            self.patch_size = (2, 2, 2)

        self.ssl_weight = float(ssl_weight)
        dim = feature_size
        encoder_out_channels = getattr(self.encoder, "out_channels", dim)
        self.z_adapter = (
            nn.Identity() if encoder_out_channels == dim else nn.Conv3d(encoder_out_channels, dim, kernel_size=1, bias=True)
        )
        self.upsample = nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 4),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 8),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 16, out_channels, kernel_size=1, stride=1),
        )

        self.choice = choice
        self.ssl_head = None
        if self.ssl_weight > 0.0:
            if text_dim is None:
                raise ValueError("text_dim must be provided when ssl_weight > 0.")
            if not hasattr(self.encoder, "skip_channels"):
                raise ValueError("Encoder must expose skip_channels when ssl_weight > 0.")
            ssl_in_dim = sum(self.encoder.skip_channels)
            self.ssl_head = self._build_ssl_head(ssl_in_dim, out_dim=text_dim, hidden_dim=2048)

    def _first_conv3d_in_channels(self, module: nn.Module) -> int:
        for m in module.modules():
            if isinstance(m, nn.Conv3d):
                return m.in_channels
        raise RuntimeError("Cannot infer decoder's first Conv3d in_channels.")

    def _pool_and_concat(self, feats):
        pooled = []
        batch_size = feats[0].shape[0]
        for feat in feats:
            pooled.append(F.adaptive_avg_pool3d(feat, (1, 1, 1)).view(batch_size, -1))
        return torch.cat(pooled, dim=1)

    def _build_ssl_head(self, in_dim, out_dim=4096, hidden_dim=2048):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x, mask, x_org, text):
        choice = self.choice
        z, skips = self.encoder(x, mask, choice)

        z = self.z_adapter(z)
        x_rec = self.upsample(z)

        voxel_mask = (
            mask.repeat_interleave(self.patch_size[0], 1)
            .repeat_interleave(self.patch_size[1], 2)
            .repeat_interleave(self.patch_size[2], 3)
            .unsqueeze(1)
            .contiguous()
        )

        loss_recon = F.l1_loss(x_org, x_rec, reduction="none")
        loss_recon = (loss_recon * voxel_mask).sum() / (voxel_mask.sum() + 1e-5) / self.in_chans

        if self.ssl_weight > 0.0:
            if skips is None:
                raise ValueError("Encoder must return skip features when ssl_weight > 0.")
            if text is None:
                raise ValueError("Text embedding is required when ssl_weight > 0.")
            feat_vec = self._pool_and_concat(skips)

            device = text.device
            if feat_vec.device != device:
                feat_vec = feat_vec.to(device, non_blocking=True)
            ssl_device = next(self.ssl_head.parameters()).device
            if ssl_device != device:
                self.ssl_head = self.ssl_head.to(device, non_blocking=True)

            proj = self.ssl_head(feat_vec)
            ssl_loss = LogRatioLoss(proj, text)
            return loss_recon + self.ssl_weight * ssl_loss
        return loss_recon

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, "no_weight_decay"):
            return {"encoder." + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, "no_weight_decay_keywords"):
            return {"encoder." + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(args):
    model_type = args.model_type
    if model_type == "unet_ssl_as_encoder":
        enc = UNetEncoderForMAE(in_chans=args.in_channels, patch_size=args.patch_size)
        model = SimMIMSkip_light(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            encoder=enc,
            choice="mae",
            ssl_weight=args.ssl_weight,
            text_dim=getattr(args, "text_dim", None),
        )
    elif model_type == "swin_skip2":
        encoder = SwinTransformerSkipForSimMIM(embed_dim=args.embed_dim)
        model = SimMIMSkip_light(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            encoder=encoder,
            feature_size=args.feature_size,
            choice="mae",
            ssl_weight=args.ssl_weight,
            text_dim=getattr(args, "text_dim", None),
        )
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    return model
