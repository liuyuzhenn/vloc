import torch
import torch.nn as nn
eps_l2_norm = 1e-10
eps_fea_norm = 1e-5

def desc_l2norm(desc):
    '''descriptors with shape NxC or NxCxHxW'''
    desc = desc / desc.pow(2).sum(dim=1, keepdim=True).add(eps_l2_norm).pow(0.5)
    return desc

class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_bias=True, is_scale=True, is_eps_leanable=False):
        """
        FRN layer as in the paper
        Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks'
        <https://arxiv.org/abs/1911.09737>
        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.is_bias = is_bias
        self.is_scale = is_scale

        self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        if self.is_scale:
            x = self.weight * x
        if self.is_bias:
            x = x + self.bias
        return x

class TLU(nn.Module):
    def __init__(self, num_features):
        """
        TLU layer as in the paper
        Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks'
        <https://arxiv.org/abs/1911.09737>
        """
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.zeros_(self.tau)
        nn.init.constant_(self.tau, -1)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)
    
class HyNet(nn.Module):
    """HyNet model definition
    """
    def __init__(self, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.2):
        super(HyNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )

    def forward(self, x, mode='eval'):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6]:
            x = layer(x)
        desc_raw = self.layer7(x).squeeze(-1).squeeze(-1)
        desc = desc_l2norm(desc_raw)

        if mode == 'train':
            return desc, desc_raw
        elif mode == 'eval':
            return desc
            
    def __str__(self) -> str:
        return 'HyNet'

class SOSNet(nn.Module):
    """
    128-dimensional SOSNet model definition trained on 32x32 patches
    """
    def __init__(self, dim_desc=128, drop_rate=0.1):
        super(SOSNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()

        self.layers = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False, eps=eps_fea_norm),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,

            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
        )

        # self.conv_head = nn.Conv2d(128, self.dim_desc, kernel_size=1, stride=1, bias=False)

        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0)
        )

        return

    def forward(self, patch):
        descr = self.desc_norm(self.layers(patch) + eps_l2_norm)
        descr = descr.view(descr.size(0), -1)
        return descr


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


# copied from torchvision\models\resnet.py#27->BasicBlock
class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


class ALNet(nn.Module):
    def __init__(self, c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128, dim: int = 128,
                 single_head: bool = True,
                 ):
        super().__init__()

        self.gate = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.block1 = ConvBlock(3, c1, self.gate, nn.BatchNorm2d)

        self.block2 = ResBlock(inplanes=c1, planes=c2, stride=1,
                               downsample=nn.Conv2d(c1, c2, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        self.block3 = ResBlock(inplanes=c2, planes=c3, stride=1,
                               downsample=nn.Conv2d(c2, c3, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        self.block4 = ResBlock(inplanes=c3, planes=c4, stride=1,
                               downsample=nn.Conv2d(c3, c4, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)

        # ================================== feature aggregation
        self.conv1 = resnet.conv1x1(c1, dim // 4)
        self.conv2 = resnet.conv1x1(c2, dim // 4)
        self.conv3 = resnet.conv1x1(c3, dim // 4)
        self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        # ================================== detector and descriptor head
        self.single_head = single_head
        if not self.single_head:
            self.convhead1 = resnet.conv1x1(dim, dim)
        self.convhead2 = resnet.conv1x1(dim, dim + 1)

    def forward(self, image):
        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool4(x2)
        x3 = self.block3(x3)  # B x c3 x H/8 x W/8
        x4 = self.pool4(x3)
        x4 = self.block4(x4)  # B x dim x H/32 x W/32

        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//8 x W//8
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//32 x W//32
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        x3_up = self.upsample8(x3)  # B x dim//4 x H x W
        x4_up = self.upsample32(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)

        # ================================== detector and descriptor head
        if not self.single_head:
            x1234 = self.gate(self.convhead1(x1234))
        x = self.convhead2(x1234)  # B x dim+1 x H x W

        descriptor_map = x[:, :-1, :, :]
        scores_map = torch.sigmoid(x[:, -1, :, :]).unsqueeze(1)

        return scores_map, descriptor_map
