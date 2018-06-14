'''Train CIFAR10 with PyTorch.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np

import torchvision
import torchvision.transforms as transforms
from load import loadCIFAR10
import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true', help='test')
args = parser.parse_args()

def adjust_lr(optimizer,epoch):
    lr=args.lr*(0.1**(epoch//30))
    if lr>=1e-6:
        for param_group in optimizer.param_groups:
            param_group['lr']=lr
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.xavier_uniform(m.bias.data)

def train(model,epoch, useCuda=True,save_point=500,modelpath='',train_loss_path='./',test_loss_path='./'):
    if useCuda:
        model = model.cuda()
    ceriation = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    step=0
    step_test=0
    train_losses = []
    test_losses = []
    for i in range(epoch):
        # trainning
        sum_loss = 0
        for batch_idx, (x, target) in enumerate(trainLoader):
            optimizer.zero_grad()
            adjust_lr(optimizer,epoch)
            if useCuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = model(x)

            loss = ceriation(out, target)
            sum_loss += loss.item()
            train_losses.append(sum_loss/(batch_idx+1))

            loss.backward()
            optimizer.step()
            step+=1
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(i, batch_idx + 1, sum_loss/(batch_idx+1)))

            if (step+1)%save_point==0:
                torch.save(model.state_dict(),modelpath)
                print("----------save finish----------------")

        #print(trainCsv)
        #train_loss_path = './save/train_loss.csv'

        # testing
        correct_cnt, sum_loss = 0, 0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(testLoader):
            x, target = Variable(x), Variable(target)
            if useCuda:
                x, target = x.cuda(), target.cuda()
            out = model(x)
            loss = ceriation(out, target)
            sum_loss+=loss.item()

            test_losses.append(sum_loss/(batch_idx+1))

            step_test+=1
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt +=(pred_label == target.data).sum()
            correct_cnt = correct_cnt.item()
            acc = (correct_cnt * 1.0 / float(total_cnt))
            # print( (correct_cnt*1.0/float(total_cnt)) )
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(testLoader):
                print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(i, batch_idx + 1,
                                                                                            sum_loss / (batch_idx + 1),
                                                                                            acc))
        trainCsv = pd.DataFrame(train_losses)
        testCsv = pd.DataFrame(test_losses)
        testCsv.to_csv(test_loss_path, index=False)
        trainCsv.to_csv(train_loss_path, index=False)
def test(model, useCuda=True):
    if useCuda:
        model = model.cuda()
    ceriation = nn.CrossEntropyLoss()
    # testing
    correct_cnt, sum_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(testLoader):
        x, target = Variable(x), Variable(target)
        if useCuda:
            x, target = x.cuda(), target.cuda()
        out = model(x)
        loss = ceriation(out, target)
        sum_loss+=loss.item()

        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        correct_cnt=correct_cnt.item()
        acc=(correct_cnt*1.0/float(total_cnt))
        #print( (correct_cnt*1.0/float(total_cnt)) )
        print('==>>> batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(batch_idx + 1, sum_loss/(batch_idx+1),acc))

import matplotlib.pyplot as plt
def plot_loss(train,test):

    plt.switch_backend('agg')
    x1=[]
    y1=[]
    for loss,index in train:
        x1.append(index)
        y1.append(loss)
    plt.plot(x1,y1,marker='*',mec='r',mfc='w')
    plt.legend()
    plt.xlabel("training epoches")
    plt.ylabel("loss")
    plt.title("training loss")
    plt.show()
    plt.savefig("./save/train_loss.jpg")
    plt.close()
    x2=[]
    y2=[]
    for loss,index in test:
        x2.append(index)
        y2.append(loss)
    plt.plot(x2,y2,"b--",linewidth=1)
    plt.xlabel("test epoches")
    plt.ylabel("loss")
    plt.title("test loss")
    plt.savefig("./save/test_loss.jpg")
    plt.close()

def plot(train_loss_path,test_loss_path):
    plt.switch_backend('agg')
    y1=pd.read_csv(train_loss_path,usecols=[0])
    x1=pd.read_csv(train_loss_path,usecols=[1])
    plt.plot(y1, x1, mec='r', mfc='w')
    plt.legend()
    plt.xlabel("training epoches")
    plt.ylabel("loss")
    plt.title("training loss")
    plt.show()
    plt.savefig("./save/train_loss.jpg")
    plt.close()

    y2 = pd.read_csv(test_loss_path, usecols=[0])
    x2 = pd.read_csv(test_loss_path, usecols=[1])
    plt.plot(y2, x2, mec='r', mfc='w')
    plt.legend()
    plt.xlabel("test epoches")
    plt.ylabel("loss")
    plt.title("test loss")
    plt.show()
    plt.savefig("./save/test_loss.jpg")
    plt.close()




if __name__ == '__main__':
    # Model
    test_loss_path = './save/test_loss.csv'
    train_loss_path = './save/train_loss.csv'
    #model_path = "/disk/yaoyang/model/new_model.pkl"
    model_path = "./save/new_model1.pkl"
    batchSize = 128
    trainLoader, testLoader = loadCIFAR10(batchSize=batchSize)
    use_cuda = torch.cuda.is_available()
    if args.test:
        print('==> loading model to test..')
        net = NewModel1()
        net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
        net.load_state_dict(torch.load(model_path))
        net = net.eval()

        cudnn.benchmark = True
        test(model=net, useCuda=use_cuda)

    else:
        if args.resume:
            print('==> loading model..')

            net = NewModel1()
            net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
            net.load_state_dict(torch.load(model_path))
            net.apply(weight_init)
        else:
            print('==> Building model..')
            #net = VGG('VGG19')
            #net = DenseNet121()
            # net = PreActResNet18()
            # net = GoogLeNet()
            #net = DenseNet121()
            net=NewModel1()
            # net = ResNeXt29_2x64d()
            # net = MobileNet()
            #net = MobileNetV2()
            # net = DPN92()
            # net = ShuffleNetG2()
            # net = SENet18()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            net.apply(weight_init)
        cudnn.benchmark = True
        #model_path="/home/zhangxu/python_project/test/save/new_model_cifar100.pt"
        #model_path = "/home/yaoyang/project/test/save/new_model_cifar100.pt"
        train(model=net, epoch=10, useCuda=use_cuda,save_point=500,modelpath=model_path,train_loss_path=train_loss_path,test_loss_path=test_loss_path)
        #plot(train_loss_path,test_loss_path)

