# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union
import math 
import torch
import torch.nn as nn
import numpy as np

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

__all__ = ["BasicUnet", "Basicunet", "basicunet", "BasicUNet"]

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


class TwoConv(nn.Sequential):
    """two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        self.temb_proj = torch.nn.Linear(512,
                                         out_chns)

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)
    
    def forward(self, x, temb):
        # print("x.shape",x.shape)  # 1 4 96 96
        x = self.conv_0(x)
        # print("conv_0.shape", x.shape)  # 1 64 96 96
        # print("temb.shape", temb.shape)  # 1 512
        # x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        temb = self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        # print("temb.shape", temb.shape)  # 1 512
        # print("type",type(temb))
        temb = temb.squeeze(-1)
        # print("temb.shape", temb.shape)  # 1 512
        # print("type", type(temb))
        # temb = nonlinearity(temb)
        # print("temb.shape", temb.shape)  # 1 512
        # temb = self.temb_proj(temb)
        # print("temb_proj.shape", temb.shape)  # 1 64
        x = x + temb
        # print("cat.shape", x.shape)  # 1, 64, 64, 96, 96
        x = self.conv_1(x)
        return x 

class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)

    def forward(self, x, temb):
        x = self.max_pooling(x)
        x = self.convs(x, temb)
        return x 

class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor], temb):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1), temb)  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0, temb)

        return x


class text_classifier(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_c, in_c // 8, bias=False), nn.ReLU(),
            nn.Linear(in_c // 8, out_c[0], bias=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_c, in_c // 8, bias=False), nn.ReLU(),
            nn.Linear(in_c // 8, out_c[1], bias=False)
        )

    def forward(self, feats):
        pool = self.avg_pool(feats).view(feats.shape[0], feats.shape[1])
        num_polyps = self.fc1(pool)
        polyp_sizes = self.fc2(pool)
        return num_polyps, polyp_sizes


class embedding_feature_fusion(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Conv2d((in_c[0] + in_c[1]) * in_c[2], out_c, 1, bias=False), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 1, bias=False), nn.ReLU()
        )

    def forward(self, num_polyps, polyp_sizes, label):
        num_polyps_prob = torch.softmax(num_polyps, axis=1)
        polyp_sizes_prob = torch.softmax(polyp_sizes, axis=1)
        prob = torch.cat([num_polyps_prob, polyp_sizes_prob], axis=1)
        prob = prob.view(prob.shape[0], prob.shape[1], 1)
        # print(label.is_cuda,prob.is_cuda)
        device = torch.device("cuda")
        label = label.to(device)
        # print(label.is_cuda, prob.is_cuda)
        # print("label.shape",label.shape)  # 1 256 256
        # print("prob.shape", prob.shape)    # 5 1
        x = label * prob
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x

class conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)

class multiscale_feature_aggregation(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8x8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.c10 = conv2d(in_c[0], out_c, kernel_size=1, padding=0)
        self.c11 = conv2d(in_c[1], out_c, kernel_size=1, padding=0)
        self.c12 = conv2d(in_c[2], out_c, kernel_size=1, padding=0)
        self.c13 = conv2d(in_c[3], out_c, kernel_size=1, padding=0)
        self.c14 = conv2d(out_c * 4, out_c, kernel_size=1, padding=0)

        self.c2 = conv2d(out_c, out_c, act=False)
        self.c3 = conv2d(out_c, out_c, act=False)

    def forward(self, x0, x1, x2, x3):
        # print("x0",x0.shape)
        x0 = self.up_8x8(x0)
        # print("x0", x0.shape)
        x1 = self.up_4x4(x1)
        # print("x1", x1.shape)
        x2 = self.up_2x2(x2)
        # print("x2", x2.shape)

        x0 = self.c10(x0)
        x1 = self.c11(x1)
        x2 = self.c12(x2)
        x3 = self.c13(x3)
        x = torch.cat([x0,x1, x2, x3], axis=1)
        x = self.c14(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        return x

class output_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        # print("x.shape",x.shape)
        # x = self.up(x)
        x = self.c1(x)
        return x


class label_attention(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        """ Channel Attention """
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c[1], in_c[0], kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_c[0], in_c[0], kernel_size=1, padding=0, bias=False)
        )

    def forward(self, feats, label):
        """ Channel Attention """
        b, c = label.shape
        label = label.reshape(b, c, 1, 1)
        ch_attn = self.c1(label)
        ch_map = torch.sigmoid(ch_attn)
        feats = feats * ch_map

        ch_attn = ch_attn.reshape(ch_attn.shape[0], ch_attn.shape[1])
        return ch_attn, feats


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.c1 = conv2d(in_c + out_c, out_c, kernel_size=1, padding=0)
        self.c2 = conv2d(out_c, out_c, act=False)
        self.c3 = conv2d(out_c, out_c, act=False)
        self.c4 = conv2d(out_c, out_c, kernel_size=1, padding=0, act=False)
        self.ca = channel_attention(out_c)
        self.sa = spatial_attention()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.c1(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        s3 = x
        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)

        x = self.ca(x)
        x = self.sa(x)
        return x


class dilated_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = nn.Sequential(conv2d(in_c, out_c, kernel_size=1, padding=0), channel_attention(out_c))
        self.c2 = nn.Sequential(conv2d(in_c, out_c, kernel_size=(3, 3), padding=6, dilation=6),
                                channel_attention(out_c))
        self.c3 = nn.Sequential(conv2d(in_c, out_c, kernel_size=(3, 3), padding=12, dilation=12),
                                channel_attention(out_c))
        self.c4 = nn.Sequential(conv2d(in_c, out_c, kernel_size=(3, 3), padding=18, dilation=18),
                                channel_attention(out_c))
        self.c5 = conv2d(out_c * 4, out_c, kernel_size=3, padding=1, act=False)
        self.c6 = conv2d(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.sa = spatial_attention()

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x1, x2, x3, x4], axis=1)
        xc = self.c5(xc)
        xs = self.c6(x)
        x = self.relu(xc + xs)
        x = self.sa(x)
        return x

import matplotlib.pyplot as plt
class BasicUNetDe(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")
        
        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])
        # spatial_dims = 2

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.upcat_4 = UpCat(spatial_dims, 128, 128, 128, act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, 128, 128, 128, act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, 128, 128, 128, act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, 128, 128, 128, act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

        self.text_classifier = text_classifier(1024, [2, 3])
        self.label_fc = embedding_feature_fusion([2, 3, 300], 128)

        """ Dilated Conv """
        self.s0 = dilated_conv(64, 128)
        self.s1 = dilated_conv(128, 128)
        self.s2 = dilated_conv(256, 128)
        self.s3 = dilated_conv(512, 128)
        self.s4 = dilated_conv(1024, 128)

        self.d0 = decoder_block(128, 128, scale=2)
        self.a0 = label_attention([128, 128])

        self.d1 = decoder_block(128, 128, scale=2)
        self.a1 = label_attention([128, 128])

        self.d2 = decoder_block(128, 128, scale=2)
        self.a2 = label_attention([128, 128])

        self.d3 = decoder_block(128, 128, scale=2)
        self.a3 = label_attention([128, 128])

        self.ag = multiscale_feature_aggregation([128,128, 128, 128], 128)

        self.y1 = output_block(128, 1)
        self.num_images = 0
    def reli(self,out):
        self.num_images += 1
        for i, f in enumerate(out):
            f = f[0].cpu().mean(dim=0)
        # if i == 4 or i == 5:
        # f = f.view(64, 64)
        path = f'fea/{self.num_images}_{i}.png'
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.imshow(f, cmap='jet')
        ax.axis('off')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()


    def forward(self,  x: torch.Tensor, t, embeddings=None, image=None,label=None):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        # print("label.shape",label.shape)
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None :
            # print("img.shape", image.shape)  # 1 3 96 96
            # print("x.shape", x.shape)  # 1 3 96 96
            x = torch.cat([image, x], dim=1)


        # print("cat.shape",x.shape) # 32 4 256 256
        # print("temb.shape",temb.shape)
        x0 = self.conv_0(x, temb)
        # print("x0.shape", x0.shape) # 32 64 256 256
        if embeddings is not None:
            x0 += embeddings[0]
            # print("emb+x0.shape", x0.shape)  # 4 64 96 96 96
        x1 = self.down_1(x0, temb)
        # print("x1.shape", x1.shape)  # 32 128 128 128
        if embeddings is not None:
            x1 += embeddings[1]
            # print("emb+x1.shape", x1.shape)   # 4 64 48 48
        x2 = self.down_2(x1, temb)
        # print("x2.shape", x2.shape)   # 32 256 64 64
        if embeddings is not None:
            x2 += embeddings[2]
            # print("emb+x2.shape", x2.shape)  # 4 128 24 24 24
        x3 = self.down_3(x2, temb)
        # print("x3.shape", x3.shape)  # 32 512 32 32
        if embeddings is not None:
            x3 += embeddings[3]
            # print("emb+x3.shape", x3.shape)  # 4 256 12 12 12
        x4 = self.down_4(x3, temb)
        # print("x4.shape", x4.shape)  # 32 1024 16 16
        if embeddings is not None:
            x4 += embeddings[4]
            # print("emb+x4.shape", x4.shape)  # 4 512 6 6 6
        num_polyps, polyp_sizes = self.text_classifier(x4)
        # print("label.shape",label.shape) # 1 128 128
        # print("num_polyps.shape", num_polyps.shape) # 2
        # print("polyp_sizes.shape", polyp_sizes.shape) # 3
        f0 = self.label_fc(num_polyps, polyp_sizes, label)

        s0 = self.s0(x0)
        s1 = self.s1(x1)
        s2 = self.s2(x2)
        s3 = self.s3(x3)
        s4 = self.s4(x4)
        out = []
        out.append(s4)
        u4 = self.upcat_4(s4, s3, temb)
        f_0, a0 = self.a0(u4, f0)

        u3 = self.upcat_3(a0, s2, temb)
        f = f0 + f_0
        f1, a1 = self.a1(u3,f)

        u2 = self.upcat_2(a1, s1, temb)
        f = f0 + f_0 + f1
        f2,a2 = self.a2(u2,f)

        u1 = self.upcat_1(a2, s0, temb)
        out.append(u1)
        f = f0 + f_0 + f1 +f2
        f3, a3 = self.a1(u1,f)
        ag = self.ag(a0,a1,a2,a3)
        y1 = self.y1(ag)
        out.append(y1)
        # print("y1.shape",y1.shape) # 2 1 512 512
        #
        # ag = self.ag(a1, a2, a3)
        # y1 = self.y1(ag)
        # print("u4.shape",u4.shape)  # 32 512 32 32
        # print("u3.shape", u3.shape)  # 32 256 64 64
        # print("u2.shape", u2.shape)  # 32 128 128 128
        # print("u1.shape", u1.shape)  # 32 128 256 256

        # logits = self.final_conv(u1)
        # print("logits",logits.shape) # 32 1 256 256
        self.reli(out)
        return y1



