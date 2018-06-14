import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()

        g = self.groups
        return x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()

        planes=4*growth_rate
        groups=2
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,groups=groups, bias=False)
        self.shuffle1=ShuffleBlock(groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes, kernel_size=3, groups=planes,padding=1, bias=False)
        self.shuffle2 = ShuffleBlock(planes)
        self.bn3=nn.BatchNorm2d(planes)
        self.conv3=nn.Conv2d(planes,growth_rate,kernel_size=1,groups=groups,bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out=self.shuffle1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.shuffle2(out)
        out = self.conv3(F.relu(self.bn3(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        #self.bn = nn.BatchNorm2d(in_planes)
        #self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

        planes=in_planes*2
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, groups=2,padding=0, bias=False)
        self.shuffle1=ShuffleBlock(2)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.shuffle2 = ShuffleBlock(planes)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out=self.shuffle1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out=self.shuffle2(out)
        out = self.conv3(F.relu(self.bn3(out)))
        return out


class NewmModel(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(NewmModel, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.linear = nn.Linear(1280, num_classes)
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def NewModel1():
    return NewmModel(Bottleneck, [5,5,3,4], growth_rate=32)

def test():
    net = NewModel1()
    x = torch.randn(1,3,32,32)
    y = net(Variable(x))
    print(y)
if __name__ == '__main__':
    test()
