'''
resnet for imgenet64*64 in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import os
from typing import Any
import torch
import torch.nn as nn
import math

__all__ = ['resnet101_t']
def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_Imagnet64X64(nn.Module):

    def __init__(self, block, layers, pretrained=None, num_classes=200):
        super(ResNet_Imagnet64X64, self).__init__()
        self.pretrained = pretrained
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # # 注意此处不让网络with_nograd()，方便精炼全网络:
        # # if self.pretrained:
        # if False:
        #     with torch.no_grad():
        #         x = self.conv1(x)
        #         x = self.bn1(x)
        #         x = self.relu(x)

        #         x = self.layer1(x)
        #         x = self.layer2(x)
        #         x = self.layer3(x)
        #         x = self.layer4(x)

        #         x = self.avgpool(x)
        #     x = x.view(x.size(0), -1)
        #     x = self.fc(x)
        
        # For kd:
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        # else:
        #     x = self.conv1(x)
        #     x = self.bn1(x)
        #     x = self.relu(x)

        #     x = self.layer1(x)
        #     x = self.layer2(x)
        #     x = self.layer3(x)
        #     x = self.layer4(x)

        #     x = self.avgpool(x)
        #     x = x.view(x.size(0), -1)
        #     x = self.fc(x)

        return x


def resnet101_t(pretrained=False, pth=None, **kwargs:Any):
    model = ResNet_Imagnet64X64(Bottleneck, [3, 4, 23, 3], **kwargs)
    '''
    先加载ImageNet训练模型微调FC层（20个epoch），
    再微调整个网络
    '''
    # if pretrained:
        # pretrained_dict = torch.load('pretrained/resnet101-5d3b4d8f.pth')
        # '''
        # for k,v in pretrained_dict.items():
        #     print(k)
        # '''
        # #删除预训练模型跟当前模型层名称相同，层结构却不同的元素;这里有两个'fc.weight'、'fc.bias'
        # pretrained_dict.pop('fc.weight')
        # pretrained_dict.pop('fc.bias')
        
        # #自己的模型参数变量
        # model_dict = model.state_dict()
        # #去除一些不需要的参数
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # #参数更新
        # model_dict.update(pretrained_dict)
        
        # # 加载我们真正需要的state_dict
        # model.load_state_dict(model_dict)
    if pretrained:
        # pth = 'pretrained/resnet101_fine_all.pth.tar'
        if not os.path.exists(pth):
            raise KeyError('pth file {} does not exist'.format(pth))
        pretrained_dict = torch.load(pth)
        model.load_state_dict(pretrained_dict['state_dict'])
        print('=> loading checkpoint "{}"'.format(pth))
    return model


# if __name__ == '__main__':
#     net = resnet29(num_classes=200)
#     y = net(torch.randn(1, 3, 64, 64))
#     print(net)
#     print(y.size())