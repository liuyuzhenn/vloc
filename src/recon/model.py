import torch
import torch.nn as nn
from torchvision.models import resnet
from copy import deepcopy
import cv2
import math
import os
from typing import Optional, Callable
eps_l2_norm = 1e-10
eps_fea_norm = 1e-5


def desc_l2norm(desc):
    '''descriptors with shape NxC or NxCxHxW'''
    desc = desc / desc.pow(2).sum(dim=1,
                                  keepdim=True).add(eps_l2_norm).pow(0.5)
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

        self.weight = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.Tensor(
            1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(
                torch.Tensor(1), requires_grad=True)
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
        self.tau = nn.parameter.Parameter(torch.Tensor(
            1, num_features, 1, 1), requires_grad=True)
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
            nn.Conv2d(32, 64, kernel_size=3, stride=2,
                      padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2,
                      padding=1, bias=is_bias),
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
            nn.LocalResponseNorm(
                2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0)
        )

        return

    def forward(self, patch):
        descr = self.desc_norm(self.layers(patch) + eps_l2_norm)
        descr = descr.view(descr.size(0), -1)
        return descr


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate=None, norm_layer=None):
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
            raise ValueError(
                'ResBlock only supports groups=1 and base_width=64')
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
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(
            scale_factor=32, mode='bilinear', align_corners=True)

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


configs = {
    'alike-t': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], '../../weights/', 'alike-t.pth')},
    'alike-s': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 96, 'dim': 96, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], '../../weights/', 'alike-s.pth')},
    'alike-n': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dim': 128, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], '../../weights/', 'alike-n.pth')},
    'alike-l': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 128, 'dim': 128, 'single_head': False, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], '../../weights/', 'alike-l.pth')},
}


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)

    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def sample_descriptor(descriptor_map, kpts, bilinear_interp=False):
    """
    :param descriptor_map: BxCxHxW
    :param kpts: list, len=B, each is Nx2 (keypoints) [h,w]
    :param bilinear_interp: bool, whether to use bilinear interpolation
    :return: descriptors: list, len=B, each is NxD
    """
    batch_size, channel, height, width = descriptor_map.shape

    descriptors = []
    for index in range(batch_size):
        kptsi = kpts[index]  # Nx2,(x,y)

        if bilinear_interp:
            descriptors_ = torch.nn.functional.grid_sample(descriptor_map[index].unsqueeze(0), kptsi.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, :, 0, :]  # CxN
        else:
            kptsi = (kptsi + 1) / 2 * \
                kptsi.new_tensor([[width - 1, height - 1]])
            kptsi = kptsi.long()
            descriptors_ = descriptor_map[index,
                                          :, kptsi[:, 1], kptsi[:, 0]]  # CxN

        descriptors_ = torch.nn.functional.normalize(descriptors_, p=2, dim=0)
        descriptors.append(descriptors_.t())

    return descriptors


class DKD(nn.Module):
    def __init__(self, radius=2, top_k=0, scores_th=0.2, n_limit=20000):
        """
        Args:
            radius: soft detection radius, kernel size is (2 * radius + 1)
            top_k: top_k > 0: return top k keypoints
            scores_th: top_k <= 0 threshold mode:  scores_th > 0: return keypoints with scores>scores_th
                                                   else: return keypoints with scores > scores.mean()
            n_limit: max number of keypoint in threshold mode
        """
        super().__init__()
        self.radius = radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        self.kernel_size = 2 * self.radius + 1
        self.temperature = 0.1  # tuned temperature
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size, padding=self.radius)

        # local xy grid
        x = torch.linspace(-self.radius, self.radius, self.kernel_size)
        # (kernel_size*kernel_size) x 2 : (w,h)
        self.hw_grid = torch.stack(torch.meshgrid(
            [x, x])).view(2, -1).t()[:, [1, 0]]

    def detect_keypoints(self, scores_map, sub_pixel=True):
        b, c, h, w = scores_map.shape
        scores_nograd = scores_map.detach()
        # nms_scores = simple_nms(scores_nograd, self.radius)
        nms_scores = simple_nms(scores_nograd, 2)

        # remove border
        nms_scores[:, :, :self.radius + 1, :] = 0
        nms_scores[:, :, :, :self.radius + 1] = 0
        nms_scores[:, :, h - self.radius:, :] = 0
        nms_scores[:, :, :, w - self.radius:] = 0

        # detect keypoints without grad
        if self.top_k > 0:
            topk = torch.topk(nms_scores.view(b, -1), self.top_k)
            indices_keypoints = topk.indices  # B x top_kmodel
        else:
            if self.scores_th > 0:
                masks = nms_scores > self.scores_th
                if masks.sum() == 0:
                    th = scores_nograd.reshape(
                        b, -1).mean(dim=1)  # th = self.scores_th
                    masks = nms_scores > th.reshape(b, 1, 1, 1)
            else:
                th = scores_nograd.reshape(
                    b, -1).mean(dim=1)  # th = self.scores_th
                masks = nms_scores > th.reshape(b, 1, 1, 1)
            masks = masks.reshape(b, -1)

            indices_keypoints = []  # list, B x (any size)
            scores_view = scores_nograd.reshape(b, -1)
            for mask, scores in zip(masks, scores_view):
                indices = mask.nonzero(as_tuple=False)[:, 0]
                if len(indices) > self.n_limit:
                    kpts_sc = scores[indices]
                    sort_idx = kpts_sc.sort(descending=True)[1]
                    sel_idx = sort_idx[:self.n_limit]
                    indices = indices[sel_idx]
                indices_keypoints.append(indices)

        keypoints = []
        scoredispersitys = []
        kptscores = []
        if sub_pixel:
            # detect soft keypoints with grad backpropagation
            patches = self.unfold(scores_map)  # B x (kernel**2) x (H*W)
            self.hw_grid = self.hw_grid.to(patches)  # to device
            for b_idx in range(b):
                patch = patches[b_idx].t()  # (H*W) x (kernel**2)
                # one dimension vector, say its size is M
                indices_kpt = indices_keypoints[b_idx]
                patch_scores = patch[indices_kpt]  # M x (kernel**2)

                # max is detached to prevent undesired backprop loops in the graph
                max_v = patch_scores.max(dim=1).values.detach()[:, None]
                # M * (kernel**2), in [0, 1]
                x_exp = ((patch_scores - max_v) / self.temperature).exp()

                # \frac{ \sum{(i,j) \times \exp(x/T)} }{ \sum{\exp(x/T)} }
                xy_residual = x_exp @ self.hw_grid / \
                    x_exp.sum(dim=1)[:, None]  # Soft-argmax, Mx2

                hw_grid_dist2 = torch.norm((self.hw_grid[None, :, :] - xy_residual[:, None, :]) / self.radius,
                                           dim=-1) ** 2
                scoredispersity = (
                    x_exp * hw_grid_dist2).sum(dim=1) / x_exp.sum(dim=1)

                # compute result keypoints
                keypoints_xy_nms = torch.stack(
                    [indices_kpt % w, indices_kpt // w], dim=1)  # Mx2
                keypoints_xy = keypoints_xy_nms + xy_residual
                keypoints_xy = keypoints_xy / keypoints_xy.new_tensor(
                    [w - 1, h - 1]) * 2 - 1  # (w,h) -> (-1~1,-1~1)

                kptscore = torch.nn.functional.grid_sample(scores_map[b_idx].unsqueeze(0),
                                                           keypoints_xy.view(
                                                               1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, 0, 0, :]  # CxN

                keypoints.append(keypoints_xy)
                scoredispersitys.append(scoredispersity)
                kptscores.append(kptscore)
        else:
            for b_idx in range(b):
                # one dimension vector, say its size is M
                indices_kpt = indices_keypoints[b_idx]
                keypoints_xy_nms = torch.stack(
                    [indices_kpt % w, indices_kpt // w], dim=1)  # Mx2
                keypoints_xy = keypoints_xy_nms / keypoints_xy_nms.new_tensor(
                    [w - 1, h - 1]) * 2 - 1  # (w,h) -> (-1~1,-1~1)
                kptscore = torch.nn.functional.grid_sample(scores_map[b_idx].unsqueeze(0),
                                                           keypoints_xy.view(
                                                               1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, 0, 0, :]  # CxN
                keypoints.append(keypoints_xy)
                scoredispersitys.append(None)
                kptscores.append(kptscore)

        return keypoints, scoredispersitys, kptscores

    def forward(self, scores_map, descriptor_map, sub_pixel=False):
        """
        :param scores_map:  Bx1xHxW
        :param descriptor_map: BxCxHxW
        :param sub_pixel: whether to use sub-pixel keypoint detection
        :return: kpts: list[Nx2,...]; kptscores: list[N,....] normalised position: -1.0 ~ 1.0
        """
        keypoints, scoredispersitys, kptscores = self.detect_keypoints(scores_map,
                                                                       sub_pixel)

        descriptors = sample_descriptor(descriptor_map, keypoints, sub_pixel)

        # keypoints: B M 2
        # descriptors: B M D
        # scoredispersitys:
        return keypoints, descriptors, kptscores, scoredispersitys


class ALike(ALNet):
    def __init__(self,
                 # ================================== feature encoder
                 c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128, dim: int = 128,
                 single_head: bool = False,
                 # ================================== detect parameters
                 radius: int = 2,
                 top_k: int = 500, scores_th: float = 0.5,
                 n_limit: int = 5000,
                 device: str = 'cpu',
                 model_path: str = ''
                 ):
        super().__init__(c1, c2, c3, c4, dim, single_head)
        self.radius = radius
        self.top_k = top_k
        self.n_limit = n_limit
        self.scores_th = scores_th
        self.dkd = DKD(radius=self.radius, top_k=self.top_k,
                       scores_th=self.scores_th, n_limit=self.n_limit)
        self.device = device

        if model_path != '':
            state_dict = torch.load(model_path, self.device)
            self.load_state_dict(state_dict)
            self.to(self.device)
            self.eval()

    def extract_dense_map(self, image, ret_dict=False):
        # ====================================================
        # check image size, should be integer multiples of 2^5
        # if it is not a integer multiples of 2^5, padding zeros
        device = image.device
        b, c, h, w = image.shape
        h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
        w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w
        if h_ != h:
            h_padding = torch.zeros(b, c, h_ - h, w, device=device)
            image = torch.cat([image, h_padding], dim=2)
        if w_ != w:
            w_padding = torch.zeros(b, c, h_, w_ - w, device=device)
            image = torch.cat([image, w_padding], dim=3)
        # ====================================================

        scores_map, descriptor_map = super().forward(image)

        # ====================================================
        if h_ != h or w_ != w:
            descriptor_map = descriptor_map[:, :, :h, :w]
            scores_map = scores_map[:, :, :h, :w]  # Bx1xHxW
        # ====================================================

        # BxCxHxW
        descriptor_map = torch.nn.functional.normalize(
            descriptor_map, p=2, dim=1)

        if ret_dict:
            return {'descriptor_map': descriptor_map, 'scores_map': scores_map, }
        else:
            return descriptor_map, scores_map

    def forward(self, img, image_size_max=99999, sort=False, sub_pixel=False):
        """
        :param img: np.array HxWx3, RGB
        :param image_size_max: maximum image size, otherwise, the image will be resized
        :param sort: sort keypoints by scores
        :param sub_pixel: whether to use sub-pixel accuracy
        :return: a dictionary with 'keypoints', 'descriptors', 'scores', and 'time'
        """
        H, W, three = img.shape
        assert three == 3, "input image shape should be [HxWx3]"

        # ==================== image size constraint
        image = deepcopy(img)
        max_hw = max(H, W)
        if max_hw > image_size_max:
            ratio = float(image_size_max / max_hw)
            image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)

        # ==================== convert image to tensor
        image = torch.from_numpy(image).to(self.device).to(
            torch.float32).permute(2, 0, 1)[None] / 255.0

        # ==================== extract keypoints
        with torch.no_grad():
            descriptor_map, scores_map = self.extract_dense_map(image)
            keypoints, descriptors, scores, _ = self.dkd(scores_map, descriptor_map,
                                                         sub_pixel=sub_pixel)
            keypoints, descriptors, scores = keypoints[0], descriptors[0], scores[0]
            keypoints = (keypoints + 1) / 2 * \
                keypoints.new_tensor([[W - 1, H - 1]])

        if sort:
            indices = torch.argsort(scores, descending=True)
            keypoints = keypoints[indices]
            descriptors = descriptors[indices]
            scores = scores[indices]
        return {'keypoints': keypoints.cpu().numpy(),
                'descriptors': descriptors.cpu().numpy(),
                'scores': scores.cpu().numpy(),
                'scores_map': scores_map.cpu().numpy(),
                }
