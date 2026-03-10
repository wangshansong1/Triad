import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from monai.utils import ensure_tuple_rep
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

class Swin(nn.Module):
    def __init__(self, args):
        super(Swin, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            use_v2=True,
        )

    def forward_encs(self, encs):
        b = encs[0].size()[0]
        outs = []
        for enc in encs:
            out = F.adaptive_avg_pool3d(enc, (1, 1, 1))
            outs.append(out.view(b, -1))
        outs = torch.cat(outs, dim=1)
        return outs

    def forward(self, x_in):
        b = x_in.size()[0]
        hidden_states_out = self.swinViT(x_in)
        out = self.forward_encs(hidden_states_out)
        return out

class TriadHead(nn.Module):
    def __init__(self, args):
        super(TriadHead, self).__init__()
        self.backbone = Swin(args)
    
    def forward(self, x):
        out = self.backbone(x)
        return out

class PlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):

        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        
    def forward_encs(self, encs):
        b = encs[0].size()[0]
        outs = []
        for enc in encs:
            out = F.adaptive_avg_pool3d(enc, (1, 1, 1))
            outs.append(out.view(b, -1))
        outs = torch.cat(outs, dim=1)
        return outs

    def forward(self, x):
        skips = self.encoder(x)
        return skips
    
def get_Plain_nnUNet(num_input_channels=1):
    UNet_base_num_features = 32
    unet_max_num_features = 320

    conv_kernel_sizes=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
    pool_op_kernel_sizes = [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
    n_conv_per_stage_encoder = [2,2,2,2,2,2]
    n_conv_per_stage_decoder = [2,2,2,2,2]

    dim = len(conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)
    num_stages = len(conv_kernel_sizes)

    conv_or_blocks_per_stage = {
                'n_conv_per_stage': n_conv_per_stage_encoder,
                'n_conv_per_stage_decoder': n_conv_per_stage_decoder
            }
    kwargs = {
                'PlainConvUNet': {
                    'conv_bias': True,
                    'norm_op': get_matching_instancenorm(conv_op),
                    'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                    'dropout_op': None, 'dropout_op_kwargs': None,
                    'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
                },
            }

    model = PlainConvUNet(
                input_channels=num_input_channels,
                n_stages=num_stages,
                features_per_stage=[min(UNet_base_num_features * 2 ** i,
                                        unet_max_num_features) for i in range(num_stages)],
                conv_op=conv_op,
                kernel_sizes=conv_kernel_sizes,
                strides=pool_op_kernel_sizes,
                **conv_or_blocks_per_stage,
                **kwargs['PlainConvUNet']
            )
    return model

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    ckpt = torch.load('Triad-PlainConvUNet-MAE.pth', weights_only=False)

    model = get_Plain_nnUNet(num_input_channels=1).to(device)
    model.load_state_dict(ckpt, strict=True)

    x = torch.randn(1, 1, 64, 160, 160, device=device)

    skips = model(x)              
    features = model.forward_encs(skips) 


    # import argparse

    # parser = argparse.ArgumentParser(description="PyTorch Training")
    # parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    # parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    # parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    # parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    # parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
    # args = parser.parse_args()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Using device:", device)

    # model = TriadHead(args).to(device)

    # ckpt = torch.load('Triad-SwinB-SimMIM.pth', weights_only=False)
    # model.load_state_dict(ckpt, strict=True)

    # x = torch.randn(1, 1, 96, 96, 96, device=device)

    # features = model(x)              
