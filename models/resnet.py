import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Encoder(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2], num_classes=10, width_per_group=64, groups=1):
        super(ResNet_Encoder, self).__init__()
        # if norm_layer is None:
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        # if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
        replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0])

        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.bn = nn.BatchNorm2d(512 * block.expansion)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # mid_out = F.relu(self.bn(out))
        # out = F.avg_pool2d(mid_out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out


class VAE_Small(nn.Module):
    def __init__(self):
        super(VAE_Small, self).__init__()

        self.fc21 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc22 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # self.fc3 = nn.Linear(z_hidden, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, in_dim)

        # Decoder
        # self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd1 = nn.ReplicationPad2d(1)
        # self.d2 = nn.Conv2d(512, 256, 3, 1)
        # self.d22 = nn.Conv2d(256, 128, 3, 1, padding=1)
        # self.bn6 = nn.BatchNorm2d(256, 1.e-3)
        # self.bn62 = nn.BatchNorm2d(128, 1.e-3)
        #
        # self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd2 = nn.ReplicationPad2d(1)
        # self.d3 = nn.Conv2d(128, 64, 3, 1)
        # self.d32 = nn.Conv2d(64, 64, 3, 1, padding=1)
        # self.bn7 = nn.BatchNorm2d(64, 1.e-3)
        # self.bn72 = nn.BatchNorm2d(64, 1.e-3)
        #
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(64, 32, 3, 1, padding=1)
        self.d42 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.bn8 = nn.BatchNorm2d(32, 1.e-3)
        self.bn82 = nn.BatchNorm2d(32, 1.e-3)

        self.fc_last = nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2, bias=False)

        # self.fc_c1 = nn.Linear(z_hidden, z_hidden)
        # self.fc_c2 = nn.Linear(z_hidden, 2)

        self.relu = nn.ReLU()

        # self.embed = nn.Embedding(10, emb_dim)

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        # h1 = F.relu(self.fc2(h1))
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))

        # h2 = F.relu(self.bn6(self.d2(self.pd1(self.up1(z)))))
        # h2 = F.relu(self.bn62(self.d22(h2)))
        # h3 = F.relu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        # h3 = F.relu(self.bn72(self.d32(h3)))
        h4 = F.relu(self.bn8(self.d4(self.up3(z))))
        h4 = F.relu(self.bn82(self.d42(self.up4(h4))))
        # h5 = F.relu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return torch.sigmoid(self.fc_last(h4))



    # def classifier(self, z):
    #     h3 = F.relu(self.fc_c1(z))
    #     return self.fc_c2(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar




class FDC_deep_preact(nn.Module):
    def __init__(self, z_hidden, sub_in_dim, block=BasicBlock, layers=[2,2,2,2], num_classes=10, width_per_group=64,
                 groups=1, zero_init_residual=True):
        super(FDC_deep_preact, self).__init__()

        self.in_planes = 64

        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.bn = nn.BatchNorm2d(512 * block.expansion)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        # if replace_stride_with_dilation is None:
        # each element in the tuple indicates if we should replace
        # the 2x2 stride with a dilated convolution instead
        replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2, bias=False) # hurt final result

        inner_dim = 128
        # self.fc_c1 = nn.Linear(z_hidden + inner_dim, z_hidden)
        self.fc_c2 = nn.Linear(z_hidden + inner_dim, 10)

        self.fc_z2 = nn.Linear(sub_in_dim, 512)
        self.fc2_z2 = nn.Linear(512, inner_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, z1, z2, test=True):
        out = self.layer2(z1)
        out = self.layer3(out)
        out = self.layer4(out)
        z1 = F.relu(self.bn(out))
        # out = F.avg_pool2d(mid_out, 4)
        # z1 = out.view(out.size(0), -1)

        # print(z1.size(), 'z1')
        bs = z1.size(0)
        z1 = F.avg_pool2d(z1, 4)
        z1 = z1.reshape(bs, -1)

        # z2 = F.relu(self.conv1(z2))
        bs_fd = z2.size(0)
        z2 = z2.reshape(bs_fd, -1)
        z2 = F.relu(self.fc_z2(z2))
        z2 = F.relu(self.fc2_z2(z2))
        # TODO: maybe add fc to z2

        if test:
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs, 1)
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs, 1, 1)

            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs * bs, -1)

            # h3 = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            out = out.view(bs, bs, -1)

            return torch.sum(out, dim=1)

        else:
            hh = torch.cat((z1, z2), dim=1)
            # h3 = F.relu(self.fc_c1(h))
            out = self.fc_c2(hh)
            return out



def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def _resnet_vissl(arch, block, layers, pretrained, progress, num_classes, **kwargs):
    model = ResNetVISSL(block, layers, num_classes=num_classes, **kwargs)
    return model

def _resnet_vissl_shallow(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetShallowVISSL(block, layers, **kwargs)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)



def resnet18vissl(pretrained=False, progress=True, num_classes=512, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_vissl('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes,
                   **kwargs)


def resnet50vissl(pretrained=False, progress=True, num_classes=1024, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_vissl('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, num_classes,
                   **kwargs)


def resnet18visslShallow(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_vissl_shallow('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

class FCClassifier(nn.Module):
    def __init__(self, out_dim=65, hidden_dim=512):
        super(FCClassifier, self).__init__()
        # self.fc_c1 = nn.Linear(512, 1024)
        self.fc_c2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, z):
        # z = F.relu(self.fc_c1(z))
        z = self.fc_c2(z)
        return z


class FC2Classifier(nn.Module):
    def __init__(self, hidden_dim=512, out_dim=65):
        super(FC2Classifier, self).__init__()
        self.fc_c1 = nn.Linear(hidden_dim, 1024)
        self.fc_c2 = nn.Linear(1024, out_dim)

    def forward(self, z):
        z = F.relu(self.fc_c1(z))
        z = self.fc_c2(z)
        return z

class FConly(nn.Module):
    def __init__(self, hidden_dim=512, out_dim=1000):
        super(FConly, self).__init__()
        self.fc_c1 = nn.Linear(hidden_dim, out_dim)

    def forward(self, z):
        z = self.fc_c1(z)
        return z

class FDC(nn.Module):
    def __init__(self, hidden_dim=2048, cat_num=65, drop_xp=False, drop_xp_ratio=0.5):
        super(FDC, self).__init__()
        sub_in_dim=64
        self.drop_xp = drop_xp

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        # self.fc_z2 = nn.Linear(sub_in_dim, 64)
        self.dropout = nn.Dropout(p=drop_xp_ratio)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.fc_c1 = nn.Linear(hidden_dim + 512, 1024)
        # self.bn = nn.BatchNorm1d(1024)
        self.fc_c2 = nn.Linear(1024, cat_num)

    def forward(self, z1, z2, test=True, random_detach=False, noise_inside=False, scale=0.01):
        z2 = F.relu(self.bn1(self.conv1(z2)))
        z2 = F.relu(self.bn2(self.conv2(z2)))
        # z2 = self.dropout(z2)
        z2 = F.relu(self.bn3(self.conv3(z2)))
        z2 = F.relu(self.bn4(self.conv4(z2)))
        z2 = F.relu(self.bn5(self.conv5(z2)))
        z2 = F.avg_pool2d(z2, 7)
        z2 = z2.reshape(z2.size(0), -1)
        if self.drop_xp:
            z2 = self.dropout(z2)
        # z2 = z2.detach()
        if random_detach:
            z2 = z2.detach()


        if test:
            # import pdb;
            # pdb.set_trace()
            bs=z1.size(0)
            bs1 = z1.size(0)
            bs2 = z2.size(0)
            zlen_1 = z1.size(1)

            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs, 1)

            if noise_inside:
                z1 = z1 + torch.normal(mean=torch.zeros((bs1, bs2, zlen_1)), std=torch.ones((bs1, bs2, zlen_1))).cuda()*scale

            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs, 1, 1)
            # z2 = torch.zeros_like(z2).cuda()

            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs*bs, -1)
            # import pdb;
            # pdb.set_trace()
            # print('h', hh.size())
            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            out = out.view(bs, bs, -1)

            return torch.sum(out, dim=1)
        else:
            # z2 = torch.zeros_like(z2).cuda()
            hh = torch.cat((z1, z2), dim=1)

            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            return out



class FDC3(nn.Module):
    def __init__(self, hidden_dim=2048, cat_num=65, drop_xp=False, drop_xp_ratio=0.5):
        super(FDC3, self).__init__()
        sub_in_dim=64
        self.drop_xp = drop_xp

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=8, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0,
                               bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0,
                               bias=False)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,
        #                        bias=False)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,
        #                        bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.fc_z2 = nn.Linear(sub_in_dim, 64)
        self.dropout = nn.Dropout(p=drop_xp_ratio)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.fc_c1 = nn.Linear(hidden_dim + 128, 1024)
        # self.bn = nn.BatchNorm1d(1024)
        self.fc_c2 = nn.Linear(1024, cat_num)

    def forward(self, z1, z2, test=True, random_detach=False):
        z2 = F.relu(self.bn1(self.conv1(z2)))
        z2 = F.relu(self.bn2(self.conv2(z2)))
        # z2 = self.dropout(z2)
        z2 = F.relu(self.bn3(self.conv3(z2)))
        # z2 = F.relu(self.bn4(self.conv4(z2)))
        # z2 = F.relu(self.bn5(self.conv5(z2)))
        z2 = F.avg_pool2d(z2, 7)
        z2 = z2.reshape(z2.size(0), -1)
        if random_detach:
            z2 = z2.detach()
        if self.drop_xp:
            z2 = self.dropout(z2)
        # z2 = z2.detach()

        if test:
            # import pdb;
            # pdb.set_trace()
            bs=z1.size(0)
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs, 1)
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs, 1, 1)
            # z2 = torch.zeros_like(z2).cuda()

            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs*bs, -1)
            # import pdb;
            # pdb.set_trace()
            # print('h', hh.size())
            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            out = out.view(bs, bs, -1)

            return torch.sum(out, dim=1)
        else:
            # z2 = torch.zeros_like(z2).cuda()
            hh = torch.cat((z1, z2), dim=1)

            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            return out



class FDC4(nn.Module):
    def __init__(self, hidden_dim=2048, cat_num=65, drop_xp=False, drop_xp_ratio=0.5):
        super(FDC4, self).__init__()
        sub_in_dim=64
        self.drop_xp = drop_xp

        self.conv1 = nn.Conv2d(3, 32, kernel_size=1, stride=4, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0,
                               bias=False)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=2, padding=0,
                               bias=False)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,
        #                        bias=False)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,
        #                        bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(256)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.fc_z2 = nn.Linear(sub_in_dim, 64)
        self.dropout = nn.Dropout(p=drop_xp_ratio)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.fc_c1 = nn.Linear(hidden_dim + 256, 1024)
        # self.bn = nn.BatchNorm1d(1024)
        self.fc_c2 = nn.Linear(1024, cat_num)

    def forward(self, z1, z2, test=True, random_detach=False):
        z2 = F.relu(self.bn1(self.conv1(z2)))
        z2 = F.relu(self.bn2(self.conv2(z2)))
        # z2 = self.dropout(z2)
        z2 = F.relu(self.bn3(self.conv3(z2)))
        # z2 = F.relu(self.bn4(self.conv4(z2)))
        # z2 = F.relu(self.bn5(self.conv5(z2)))
        z2 = F.avg_pool2d(z2, 14)
        z2 = z2.reshape(z2.size(0), -1)
        if random_detach:
            z2 = z2.detach()
        if self.drop_xp:
            z2 = self.dropout(z2)
        # z2 = z2.detach()

        if test:
            # import pdb;
            # pdb.set_trace()
            bs=z1.size(0)
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs, 1)
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs, 1, 1)
            # z2 = torch.zeros_like(z2).cuda()

            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs*bs, -1)
            # import pdb;
            # pdb.set_trace()
            # print('h', hh.size())
            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            out = out.view(bs, bs, -1)

            return torch.sum(out, dim=1)
        else:
            # z2 = torch.zeros_like(z2).cuda()
            hh = torch.cat((z1, z2), dim=1)

            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            return out



class FDC5(nn.Module):
    def __init__(self, hidden_dim=2048, cat_num=65, drop_xp=False, drop_xp_ratio=0.5, middle_hidden=1024):
        super(FDC5, self).__init__()
        sub_in_dim=64
        print("FDC 5")
        self.drop_xp = drop_xp

        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=4, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0,
                               bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0,
                               bias=False)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,
        #                        bias=False)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,
        #                        bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.fc_z2 = nn.Linear(sub_in_dim, 64)
        self.dropout = nn.Dropout(p=drop_xp_ratio)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.fc_c1 = nn.Linear(hidden_dim + 128, middle_hidden)
        # self.bn = nn.BatchNorm1d(1024)
        self.fc_c2 = nn.Linear(middle_hidden, cat_num)

    def forward(self, z1, z2, test=True, random_detach=False, noise_inside=False, scale=0.01):
        z2 = F.relu(self.bn1(self.conv1(z2)))
        z2 = F.relu(self.bn2(self.conv2(z2)))
        # z2 = self.dropout(z2)
        z2 = F.relu(self.bn3(self.conv3(z2)))
        # z2 = F.relu(self.bn4(self.conv4(z2)))
        # z2 = F.relu(self.bn5(self.conv5(z2)))
        z2 = F.avg_pool2d(z2, 14)
        z2 = z2.reshape(z2.size(0), -1)
        if random_detach:
            z2 = z2.detach()
        if self.drop_xp:
            z2 = self.dropout(z2)
        # z2 = z2.detach()

        if test:
            # import pdb;
            # pdb.set_trace()
            # Pairwise combination betwen feature from Image and feature from random image.
            # with larger batch size for z2, you get better randomized

            bs1=z1.size(0)
            bs2=z2.size(0)
            zlen_1=z1.size(1)
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs2, 1)

            if noise_inside:
                z1 = z1 + torch.normal(mean=torch.zeros((bs1, bs2, zlen_1)), std=torch.ones((bs1, bs2, zlen_1))).cuda()*scale
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs1, 1, 1)
            # z2 = torch.zeros_like(z2).cuda()
            # print(z1.shape, z2.shape)
            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs1*bs2, -1)
            # import pdb;
            # pdb.set_trace()
            # print('h', hh.size())
            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            out = out.view(bs1, bs2, -1)

            return torch.sum(out, dim=1)
        else:
            # z2 = torch.zeros_like(z2).cuda()
            hh = torch.cat((z1, z2), dim=1)

            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            return out



class FDC5_PyzxOverfit(nn.Module):
    def __init__(self, hidden_dim=2048, cat_num=65, drop_xp=False, drop_xp_ratio=0.5, middle_hidden=1024):
        super(FDC5_PyzxOverfit, self).__init__()
        sub_in_dim=64
        print("FDC 5")
        self.drop_xp = drop_xp

        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=4, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0,
                               bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0,
                               bias=False)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,
        #                        bias=False)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,
        #                        bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.fc_z2 = nn.Linear(sub_in_dim, 64)
        self.dropout = nn.Dropout(p=drop_xp_ratio)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.fc_c1 = nn.Linear(hidden_dim + 128, middle_hidden)
        self.fc_cadd = nn.Linear(middle_hidden, middle_hidden)
        # self.bn = nn.BatchNorm1d(1024)
        self.fc_c2 = nn.Linear(middle_hidden, cat_num)

    def forward(self, z1, z2, test=True, random_detach=False, noise_inside=False, scale=0.01):
        z2 = F.relu(self.bn1(self.conv1(z2)))
        z2 = F.relu(self.bn2(self.conv2(z2)))
        # z2 = self.dropout(z2)
        z2 = F.relu(self.bn3(self.conv3(z2)))
        # z2 = F.relu(self.bn4(self.conv4(z2)))
        # z2 = F.relu(self.bn5(self.conv5(z2)))
        z2 = F.avg_pool2d(z2, 14)
        z2 = z2.reshape(z2.size(0), -1)
        if random_detach:
            z2 = z2.detach()
        if self.drop_xp:
            z2 = self.dropout(z2)
        # z2 = z2.detach()

        if test:
            # import pdb;
            # pdb.set_trace()
            # Pairwise combination betwen feature from Image and feature from random image.
            # with larger batch size for z2, you get better randomized

            bs1=z1.size(0)
            bs2=z2.size(0)
            zlen_1=z1.size(1)
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs2, 1)

            if noise_inside:
                z1 = z1 + torch.normal(mean=torch.zeros((bs1, bs2, zlen_1)), std=torch.ones((bs1, bs2, zlen_1))).cuda()*scale
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs1, 1, 1)
            # z2 = torch.zeros_like(z2).cuda()
            # print(z1.shape, z2.shape)
            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs1*bs2, -1)
            # import pdb;
            # pdb.set_trace()
            # print('h', hh.size())
            hh = F.relu(self.fc_c1(hh))
            hh = F.relu(self.fc_cadd(hh))
            out = self.fc_c2(hh)
            out = out.view(bs1, bs2, -1)

            return torch.sum(out, dim=1)
        else:
            # z2 = torch.zeros_like(z2).cuda()
            hh = torch.cat((z1, z2), dim=1)

            hh = F.relu(self.fc_c1(hh))
            hh = F.relu(self.fc_cadd(hh))
            out = self.fc_c2(hh)
            return out



class FDC5_ada(nn.Module):
    def __init__(self, hidden_dim=2048, cat_num=65, drop_xp=False, drop_xp_ratio=0.5, middle_hidden=1024, bddim=128):
        super(FDC5_ada, self).__init__()
        sub_in_dim=64
        print("FDC 5")
        self.drop_xp = drop_xp

        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=4, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0,
                               bias=False)
        self.conv3 = nn.Conv2d(64, bddim, kernel_size=1, stride=2, padding=0,
                               bias=False)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,
        #                        bias=False)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,
        #                        bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(bddim)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.fc_z2 = nn.Linear(sub_in_dim, 64)
        self.dropout = nn.Dropout(p=drop_xp_ratio)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.fc_c1 = nn.Linear(hidden_dim + bddim, middle_hidden)
        # self.bn = nn.BatchNorm1d(1024)
        self.fc_c2 = nn.Linear(middle_hidden, cat_num)

    def forward(self, z1, z2, test=True, random_detach=False):
        z2 = F.relu(self.bn1(self.conv1(z2)))
        z2 = F.relu(self.bn2(self.conv2(z2)))
        # z2 = self.dropout(z2)
        z2 = F.relu(self.bn3(self.conv3(z2)))
        # z2 = F.relu(self.bn4(self.conv4(z2)))
        # z2 = F.relu(self.bn5(self.conv5(z2)))
        z2 = F.avg_pool2d(z2, 14)
        z2 = z2.reshape(z2.size(0), -1)
        if random_detach:
            z2 = z2.detach()
        if self.drop_xp:
            z2 = self.dropout(z2)
        # z2 = z2.detach()

        if test:
            # import pdb;
            # pdb.set_trace()
            bs1=z1.size(0)
            bs2=z2.size(0)
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs2, 1)
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs1, 1, 1)
            # z2 = torch.zeros_like(z2).cuda()
            # print(z1.shape, z2.shape)
            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs1*bs2, -1)
            # import pdb;
            # pdb.set_trace()
            # print('h', hh.size())
            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            out = out.view(bs1, bs2, -1)

            return torch.sum(out, dim=1)
        else:
            # z2 = torch.zeros_like(z2).cuda()
            hh = torch.cat((z1, z2), dim=1)

            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            return out



class FDC5Linear(nn.Module):
    def __init__(self, hidden_dim=2048, cat_num=65, drop_xp=False, drop_xp_ratio=0.5, middle_hidden=1024, bd_dim=32):
        super(FDC5Linear, self).__init__()
        sub_in_dim=64
        print("FDC 5")
        self.drop_xp = drop_xp

        bd_dim=bd_dim

        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=4, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0,
                               bias=False)
        self.conv3 = nn.Conv2d(64, bd_dim, kernel_size=1, stride=2, padding=0,
                               bias=False)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,
        #                        bias=False)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,
        #                        bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(bd_dim)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.fc_z2 = nn.Linear(sub_in_dim, 64)
        self.dropout = nn.Dropout(p=drop_xp_ratio)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.fc_c1 = nn.Linear(hidden_dim + bd_dim, cat_num)

    def forward(self, z1, z2, test=True, random_detach=False):
        z2 = F.relu(self.bn1(self.conv1(z2)))
        z2 = F.relu(self.bn2(self.conv2(z2)))
        # z2 = self.dropout(z2)
        z2 = F.relu(self.bn3(self.conv3(z2)))
        # z2 = F.relu(self.bn4(self.conv4(z2)))
        # z2 = F.relu(self.bn5(self.conv5(z2)))
        z2 = F.avg_pool2d(z2, 14)
        z2 = z2.reshape(z2.size(0), -1)
        if random_detach:
            z2 = z2.detach()
        if self.drop_xp:
            z2 = self.dropout(z2)
        # z2 = z2.detach()

        if test:
            # import pdb;
            # pdb.set_trace()
            bs1=z1.size(0)
            bs2=z2.size(0)
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs2, 1)
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs1, 1, 1)
            # z2 = torch.zeros_like(z2).cuda()
            # print(z1.shape, z2.shape)
            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs1*bs2, -1)
            # import pdb;
            # pdb.set_trace()
            # print('h', hh.size())
            # hh = F.relu(self.fc_c1(hh))
            out = self.fc_c1(hh)
            out = out.view(bs1, bs2, -1)

            return torch.sum(out, dim=1)
        else:
            # z2 = torch.zeros_like(z2).cuda()
            hh = torch.cat((z1, z2), dim=1)

            # hh = F.relu(self.fc_c1(hh))
            out = self.fc_c1(hh)
            return out


class FDC_I9(nn.Module):
    def __init__(self, hidden_dim=2048, cat_num=65, drop_xp=False, drop_xp_ratio=0.5, middle_hidden=1024, bddim=128, ker=4):
        super(FDC_I9, self).__init__()
        sub_in_dim=64
        print("FDC 5")
        self.drop_xp = drop_xp

        self.conv1 = nn.Conv2d(3, 32, kernel_size=ker, stride=4, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0,
                               bias=False)
        self.conv3 = nn.Conv2d(64, bddim, kernel_size=1, stride=2, padding=0,
                               bias=False)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,
        #                        bias=False)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,
        #                        bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(bddim)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.fc_z2 = nn.Linear(sub_in_dim, 64)
        self.dropout = nn.Dropout(p=drop_xp_ratio)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.pool_avg = nn.AdaptiveAvgPool2d((1,1))
        self.pool_avg2 = nn.AdaptiveAvgPool2d((1,1))

        self.fc_c1 = nn.Linear(hidden_dim + bddim, middle_hidden)
        # self.bn = nn.BatchNorm1d(1024)
        self.fc_c2 = nn.Linear(middle_hidden, cat_num)

    def forward(self, z1, z2, test=True, random_detach=False):
        z2 = F.relu(self.bn1(self.conv1(z2)))
        z2 = F.relu(self.bn2(self.conv2(z2)))
        # z2 = self.dropout(z2)
        z2 = F.relu(self.bn3(self.conv3(z2)))
        # z2 = F.relu(self.bn4(self.conv4(z2)))
        # z2 = F.relu(self.bn5(self.conv5(z2)))
        z2_mean = self.pool_avg(z2)
        z2 = self.pool_avg((z2-z2_mean)**2) # variance of features
        z2 = z2.reshape(z2.size(0), -1)
        if random_detach:
            z2 = z2.detach()
        if self.drop_xp:
            z2 = self.dropout(z2)
        # z2 = z2.detach()

        if test:
            # import pdb;
            # pdb.set_trace()
            bs1=z1.size(0)
            bs2=z2.size(0)
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs2, 1)
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs1, 1, 1)
            # z2 = torch.zeros_like(z2).cuda()
            # print(z1.shape, z2.shape)
            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs1*bs2, -1)
            # import pdb;
            # pdb.set_trace()
            # print('h', hh.size())
            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            out = out.view(bs1, bs2, -1)

            return torch.sum(out, dim=1)
        else:
            # z2 = torch.zeros_like(z2).cuda()
            hh = torch.cat((z1, z2), dim=1)

            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            return out

class FDC6(nn.Module):
    def __init__(self, hidden_dim=2048, cat_num=65, drop_xp=False, drop_xp_ratio=0.5):
        super(FDC6, self).__init__()
        sub_in_dim=64
        self.drop_xp = drop_xp

        self.conv1 = nn.Conv2d(3, 32, kernel_size=1, stride=4, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0,
                               bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0,
                               bias=False)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,
        #                        bias=False)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,
        #                        bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.fc_z2 = nn.Linear(sub_in_dim, 64)
        self.dropout = nn.Dropout(p=drop_xp_ratio)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.fc_c1 = nn.Linear(hidden_dim + 128, 1024)
        # self.bn = nn.BatchNorm1d(1024)
        self.fc_c2 = nn.Linear(1024, cat_num)

    def forward(self, z1, z2, test=True, random_detach=False):
        z2 = F.relu(self.bn1(self.conv1(z2)))
        z2 = F.relu(self.bn2(self.conv2(z2)))
        # z2 = self.dropout(z2)
        z2 = F.relu(self.bn3(self.conv3(z2)))
        # z2 = F.relu(self.bn4(self.conv4(z2)))
        # z2 = F.relu(self.bn5(self.conv5(z2)))
        z2 = F.avg_pool2d(z2, 14)
        z2 = z2.reshape(z2.size(0), -1)
        if random_detach:
            z2 = z2.detach()
        if self.drop_xp:
            z2 = self.dropout(z2)
        # z2 = z2.detach()

        if test:
            # import pdb;
            # pdb.set_trace()
            bs=z1.size(0)
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs, 1)
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs, 1, 1)
            # z2 = torch.zeros_like(z2).cuda()

            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs*bs, -1)
            # import pdb;
            # pdb.set_trace()
            # print('h', hh.size())
            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            out = out.view(bs, bs, -1)

            return torch.sum(out, dim=1)
        else:
            # z2 = torch.zeros_like(z2).cuda()
            hh = torch.cat((z1, z2), dim=1)

            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            return out


class FDC2(nn.Module):
    def __init__(self, hidden_dim=2048):
        super(FDC2, self).__init__()
        sub_in_dim=64

        self.project = nn.Linear(3*56*56, 1024)

        # self.fc_z2 = nn.Linear(sub_in_dim, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc_c2 = nn.Linear(hidden_dim + 1024, 65)

    def forward(self, z1, z2, test=True):
        # z2 = F.relu(self.conv1(z2))
        # z2 = F.relu(self.conv2(z2))
        # z2 = self.dropout(z2)
        # z2 = F.relu(self.conv3(z2))
        # z2 = F.avg_pool2d(z2, 14)
        # z2 = z2.reshape(z2.size(0), -1)
        # z2 = self.dropout(z2)
        z2 = z2.reshape(z2.size(0), -1)
        z2 = F.relu(self.project(z2))
        z2 = z2.detach()

        if test:
            # import pdb;
            # pdb.set_trace()
            bs=z1.size(0)
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs, 1)
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs, 1, 1)

            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs*bs, -1)
            # import pdb;
            # pdb.set_trace()
            # print('h', hh.size())
            # h3 = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            out = out.view(bs, bs, -1)

            return torch.sum(out, dim=1)
        else:
            hh = torch.cat((z1, z2), dim=1)
            # h3 = F.relu(self.fc_c1(h))
            out = self.fc_c2(hh)
            return out



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, nofc=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.nofc = nofc

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if nofc ==False:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        hidden = torch.flatten(x, 1)
        # hidden = self.dropout(hidden)
        if self.nofc == False:
            x = self.fc(hidden)

        return x, hidden


class ResNetVISSL(nn.Module):
    def __init__(self, block, layers, num_classes=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetVISSL, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        hidden = torch.flatten(x, 1)

        return self.fc1(hidden), self.fc2(hidden)




class ResNetShallowVISSL(nn.Module):
    def __init__(self, block, layers, num_classes=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetShallowVISSL, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.fc2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.dropout = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # hidden = torch.flatten(x, 1)

        return self.fc1(x), self.fc2(x)


class FDC5Shallow(nn.Module):
    def __init__(self, hidden_dim=2048, cat_num=65, drop_xp=False, drop_xp_ratio=0.5):
        super(FDC5Shallow, self).__init__()
        sub_in_dim=64
        print("FDC 5")
        self.drop_xp = drop_xp
        block = BasicBlock
        layers=[2, 2, 2, 2]
        self.dilation = 1
        self.groups = 1
        self.base_width=64

        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=4, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0,
                               bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0,
                               bias=False)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,
        #                        bias=False)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,
        #                        bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.fc_z2 = nn.Linear(sub_in_dim, 64)
        self.dropout = nn.Dropout(p=drop_xp_ratio)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.inplanes = 256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_c1 = nn.Linear(hidden_dim + 128, 1024)
        # self.bn = nn.BatchNorm1d(1024)
        self.fc_c2 = nn.Linear(1024, cat_num)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, z1, z2, test=True, random_detach=False):
        z1 = self.layer4(z1)
        z1 = self.avgpool(z1)
        z1 = torch.flatten(z1, 1)
        latent = z1

        z2 = F.relu(self.bn1(self.conv1(z2)))
        z2 = F.relu(self.bn2(self.conv2(z2)))
        # z2 = self.dropout(z2)
        z2 = F.relu(self.bn3(self.conv3(z2)))
        # z2 = F.relu(self.bn4(self.conv4(z2)))
        # z2 = F.relu(self.bn5(self.conv5(z2)))
        z2 = F.avg_pool2d(z2, 14)
        z2 = z2.reshape(z2.size(0), -1)
        if random_detach:
            z2 = z2.detach()
        if self.drop_xp:
            z2 = self.dropout(z2)
        # z2 = z2.detach()

        if test:
            # import pdb;
            # pdb.set_trace()
            bs=z1.size(0)
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs, 1)
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs, 1, 1)
            # z2 = torch.zeros_like(z2).cuda()

            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs*bs, -1)
            # import pdb;
            # pdb.set_trace()
            # print('h', hh.size())
            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            out = out.view(bs, bs, -1)

            return torch.sum(out, dim=1)
        else:
            # z2 = torch.zeros_like(z2).cuda()
            hh = torch.cat((z1, z2), dim=1)

            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            return out, latent


