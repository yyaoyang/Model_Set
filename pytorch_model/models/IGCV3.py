import torch.nn as nn
import torch.functional as F
import math

def conv_bn(inp,oup,stride):
    return nn.Sequential(
        nn.Conv2d(inp,oup,kernel_size=3,stride=stride,padding=1,bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp,oup):
    return nn.Sequential(
        nn.Conv2d(inp,oup,kernel_size=1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class PermutationBlock(nn.Module):
    def __init__(self,groups):
        super(PermutationBlock, self).__init__()
        self.groups=groups
    def forward(self,input):
        n,c,h,w=input.size()
        G=self.groups
        output=input.view(n,G,c//G,h,w).permute(0,2,1,3,4).contiguous().view(n,c,h,w)
        return output

class InvertedResidual(nn.Module):
    def __init__(self,inp,oup,stride,expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride=stride
        assert stride in [1,2]

        self.use_res_connect=self.stride==1 and inp==oup

        self.conv=nn.Sequential(
            #pw
            nn.Conv2d(inp,inp*expand_ratio,kernel_size=1,stride=1,padding=0,groups=2,bias=False),
            nn.BatchNorm2d(inp*expand_ratio),
            nn.ReLU6(inplace=True),
            #permutation
            PermutationBlock(groups=2),
            #dw
            nn.Conv2d(inp*expand_ratio,inp*expand_ratio,kernel_size=3,stride=stride,padding=1,groups=inp*expand_ratio,bias=False),
            nn.BatchNorm2d(inp*expand_ratio),
            nn.ReLU6(inplace=True),
            #pw_linear
            nn.Conv2d(inp*expand_ratio,oup,kernel_size=1,stride=1,padding=0,groups=2,bias=False),
            nn.BatchNorm2d(oup),
            #permutation
            PermutationBlock(groups=int(round(oup/2)))
        )
    def forward(self,x):
        if self.use_res_connect:
            return x+self.conv(x)
        else:
            return self.conv(x)

class IGCV3(nn.Module):
    def __init__(self,args):
        super(IGCV3, self).__init__()
        s1,s2=2,2
        if args.downsampling==16:
            s1,s2=2,1
        elif args.downsampling==8:
            s1,s2=1,1
