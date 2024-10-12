import torch
from torch import Tensor
from torch.nn import Module, Conv1d, Identity, Linear, GELU, Sequential, ModuleList, Parameter, BatchNorm1d
import torch.nn.functional as F
import torch.nn as nn

from helper_modules import Permute, Scaler, DropPath, trunc_normal_, GradientReversal

# 1D reimplementation of the ConvNext model available at https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
class Block(Module):
    def __init__(self, dim: int, kernel_size:int = 7,
                 drop_path: float = 0.0, layer_scale_init_value: float=1e-6):
        super().__init__()
        self.block = Sequential(
            Conv1d(dim, dim, kernel_size, padding='same', groups=dim),
            Permute((0, 2, 1)),
            LayerNorm(dim, eps=1e-6),
            Linear(dim, 4 * dim),
            GELU(),
            Linear(4 * dim, dim),
            Scaler(dim, layer_scale_init_value),
            Permute((0, 2, 1))
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else Identity()
        
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x) + self.drop_path(x)

class ConvNeXt1D(Module):
    def __init__(self, in_chans=3, num_classes=20, num_subjects=35,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 return_embedding: bool = False):
        super().__init__()
        self.emb = return_embedding

        self.downsample_layers = ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = Sequential(
            BatchNorm1d(in_chans),
            Conv1d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6) # final norm layer
        
        # Domain adaptation
        self.head_subject = Sequential(
            GradientReversal(alpha=1.0),
            Linear(dims[-1], dims[-1]),
            GELU(),
            Linear(dims[-1], num_subjects)
        )
        
        # Class prediction for constraining
        self.head_classes = Sequential(
            Linear(dims[-1], dims[-1]),
            GELU(),
            Linear(dims[-1], num_classes)
        )

        self.apply(self._init_weights)
        """
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        """

    def _init_weights(self, m):
        if isinstance(m, (Conv1d, Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean(-1)) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        e = self.forward_features(x)
        xs = self.head_subject(e)
        xc = self.head_classes(e)
        return e, xs, xc


class LayerNorm(Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x: Tensor):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x

if __name__ == "__main__":
    import time
    x = torch.zeros((16, 19, 2048)).float().cuda()
    net = ConvNeXt1D(in_chans=19).float().cuda()

    t0 = time.time()
    n_repeats = 25
    for i in range(n_repeats):
        y = net(x)
    print(f"{(time.time() - t0) / n_repeats:.5f} s average per signal {x.shape}")
    