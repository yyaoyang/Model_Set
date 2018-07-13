import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np
class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups

        out=x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

        return out

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate,groups=4):
        super(Bottleneck, self).__init__()

        self.groups=groups
        self.planes=self.groups*growth_rate
        self.out_planes=growth_rate
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, self.planes, kernel_size=1,groups=self.groups, bias=False)
        self.shuffle1=ShuffleBlock(self.groups)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.conv2 = nn.Conv2d(self.planes,self.planes, kernel_size=3, groups=self.groups,padding=1, bias=False)
        self.shuffle2 = ShuffleBlock(self.groups)
        self.bn3=nn.BatchNorm2d(self.groups)
        self.conv3=nn.Conv2d(self.groups,self.out_planes,kernel_size=1,groups=self.groups,bias=False)
        self.shuffle3 = ShuffleBlock(self.groups)

        # SE layers
        self.fc1 = nn.Conv2d(self.out_planes, self.out_planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(self.out_planes // 16, self.out_planes, kernel_size=1)


    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))

        # out = torch.split(out, self.planes, 1)
        # out=torch.max(out[0],out[1])
        out=self.shuffle1(out)

        out = self.conv2(F.relu(self.bn2(out)))
        out = self.shuffle2(out)
        out=torch.split(out,self.groups,1)
        out=torch.max(out[0],out[1])


        out = self.conv3(F.relu(self.bn3(out)))

        # out = torch.split(out, self.planes, 1)
        # out = torch.max(out[0],out[1])
        out = self.shuffle3(out)

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        #out = out * w


        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        #self.bn = nn.BatchNorm2d(in_planes)
        #self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

        self.planes=in_planes*2
        self.out_planes=out_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, self.planes, kernel_size=1, stride=1, groups=2,padding=0, bias=False)
        self.shuffle1=ShuffleBlock(2)

        self.bn2 = nn.BatchNorm2d(self.planes)
        self.conv2 = nn.Conv2d(self.planes, self.planes, kernel_size=3, stride=2, padding=1, groups=self.planes, bias=False)
        self.shuffle2 = ShuffleBlock(self.planes)

        self.bn3 = nn.BatchNorm2d(self.planes)
        self.conv3 = nn.Conv2d(self.planes, self.out_planes, kernel_size=1,groups=2, stride=1, padding=0, bias=False)
        self.shuffle3 = ShuffleBlock(2)


    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        # out1 = torch.split(out, self.planes, 1)
        # max_out1 = torch.max(out1[0],out1[1])
        out = self.shuffle1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        #out=self.shuffle2(out)
        out = self.conv3(F.relu(self.bn3(out)))
        # out2 = torch.split(out, self.out_planes,1)
        # max_out2 = torch.max(out2[0],out2[1])
        out = self.shuffle3(out)

        return out


class NewmModel(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(NewmModel, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        #self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, stride=2, bias=False)

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

        #self.linear = nn.Linear(1280, num_classes)
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
        out=F.avg_pool2d(out,3,2)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    # def save(self, checkpointFold, epoch):
    #     filename = '%s/jps_%03i.pth.tar' % (checkpointFold, epoch)
    #     torch.save(self.state_dict(), filename)


def NewModel1(num_classes=10):#19
    return NewmModel(Bottleneck, [3,5,8,3], growth_rate=32,num_classes=num_classes)

def NewModel24(num_classes=10):
    return NewmModel(Bottleneck, [3,6,12,3], growth_rate=32,num_classes=num_classes)

def test():
    net = NewModel1()
    x = torch.randn(1,3,224,224)
    y = net(Variable(x))
    print(y)
if __name__ == '__main__':
    test()
