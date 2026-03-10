from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from torch import nn
import torch
from typing import Union, Type, List, Tuple
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class projection_head(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, input):
        if torch.is_tensor(input):
            x = input
        else:
            x = input[-1]
            b = x.size()[0]
            x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(b, -1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
    
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
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
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

        in_dim = 1120
        hidden_dim, out_dim = 768, 768
        self.logratio_head = projection_head(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)

        dim = features_per_stage[-1]
        self.conv = nn.Sequential(
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
                nn.Conv3d(dim // 16, input_channels, kernel_size=1, stride=1),
            )
        
    def forward_encs(self, encs):
        b = encs[0].size()[0]
        outs = []
        for enc in encs:
            out = F.adaptive_avg_pool3d(enc, (1, 1, 1))
            outs.append(out.view(b, -1))
        outs = torch.cat(outs, dim=1)
        return outs

    def forward(self, x, des, downstream):
        if not downstream:
            skips = self.encoder(x)

            x_out = self.forward_encs(skips)
            x_out = nn.Dropout1d(0.2)(x_out)
            x_out = self.logratio_head(x_out)
            logratioloss = LogRatioLoss(x_out, des)

            dec = nn.Dropout3d(0.2)(skips[-1])
            x_rec = self.conv(dec)
            rec_loss = RECLoss(x_rec, x)

            loss = rec_loss + 0.01 * logratioloss

            return loss
        else:
            skips = self.encoder(x)

            dec = nn.Dropout3d(0.2)(skips[-1])
            x_rec = self.conv(dec)
            rec_loss = RECLoss(x_rec, x)

            return rec_loss



    

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


def L2dist(x1, x2):
    eps = 1e-4 / x1.size(0)
    diff = torch.abs(x1 - x2)
    out = torch.pow(diff, 2).sum(dim=1)
    return torch.pow(out + eps, 1. / 2)

def LogRatioLoss(input, gt_dist):
    shuffled_1 = torch.randperm(input.size(0))
    shuffled_2 = torch.randperm(input.size(0))

    f_i = input[shuffled_1]
    f_j = input[shuffled_2]

    y_i = gt_dist[shuffled_1]
    y_j = gt_dist[shuffled_2]

    epsilon = 1e-6

    Dfai = L2dist(input,f_i)
    Dfaj = L2dist(input,f_j)

    Dyai = L2dist(gt_dist,y_i)
    Dyaj = L2dist(gt_dist,y_j)

    log_dist = torch.log(Dfai/Dfaj + epsilon)
    log_gt_dist = torch.log(Dyai/Dyaj + epsilon)

    log_ratio_loss = (log_dist-log_gt_dist).pow(2)

    return log_ratio_loss.mean()

def RECLoss(output_recons, target_recons):
    return F.l1_loss(output_recons, target_recons)


if __name__ == '__main__':

    from monai.networks.nets.vit import ViT
    import torch

    # Instantiate the 3D Vision Transformer model
    model = ViT(
        in_channels=1,            # Number of input channels (e.g., grayscale image)
        img_size=(96, 96, 96),    # Input image size (3D)
        patch_size=(16, 16, 16),  # Size of each patch
        hidden_size=768,          # Hidden size of the transformer (ViT/B configuration)
        mlp_dim=3072,             # MLP size in transformer layers
        num_layers=12,            # Number of transformer layers
        num_heads=12,             # Number of attention heads
    )


    # Compute the total number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    


    # num_input_channels = 1
    # model = get_Plain_nnUNet(num_input_channels)
    # pytorch_total_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)

    # print("Total parameters count", pytorch_total_params)
    # try:
    #     model_dict = torch.load("./checkpoint_final.pth", map_location=torch.device('cpu'))['network_weights']
    #     print(model_dict.keys())

    #     current_model_dict = model.state_dict()
    #     new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
    #                       zip(current_model_dict.keys(), model_dict.values())}
    #     print(new_state_dict.keys())

    #     model.load_state_dict(new_state_dict, strict=True)
    #     print("Using pretrained nnUNet weights !")

    #     # torch.save(model_dict, 'model.pth')
    #     # torch.save(new_state_dict, 'new_model.pth')

    # except ValueError:
    #     raise ValueError("Self-supervised pre-trained weights not available")

    # x = torch.rand([2, num_input_channels, 96, 96, 96])
    # des = torch.rand([2, 768])
    # y = model(x, des)
    # print(y[0].shape)
