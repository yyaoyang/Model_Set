'''Train CIFAR10 with PyTorch.'''

import argparse

import pandas as pd
import torch.optim as optim
from torch.autograd import Variable

from load import *
from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true', help='test')
parser.add_argument('--data', '-d',default='cifar10',type=str, help='load_data')
parser.add_argument('--save', '-s',default='new_model',type=str, help='save_model')
args = parser.parse_args()

def adjust_lr(optimizer,epoch,lamda=60):
    lr = args.lr*(0.1**(epoch//lamda))
    if lr>=1e-6:
        for param_group in optimizer.param_groups:
            param_group['lr']=lr

def load_data(batchSize):
    if args.data=='cifar10':
        print('loading cifar10----------------')
        trainLoader, testLoader = loadCIFAR10(batchSize=batchSize)
    elif args.data=='cifar100':
        print('loading cifar100----------------')
        trainLoader, testLoader = loadCIFAR100(batchSize=batchSize)
    elif args.data=='pascal':
        print('loading pascal----------------')
        trainLoader, testLoader = load_PASCAL(batchsize=20)
    return trainLoader,testLoader

def train(model,epoch,checkPoint,savePoint,modelPath,curEpoch=0,best_acc=0,useCuda=True,
          adjustLR=True,earlyStop=True,tolearnce=4):
    tolearnce_cnt=0
    train_loss=[]
    step=0
    if useCuda:
        model=model.cuda()
    ceritation=nn.CrossEntropyLoss()
    optimizier=optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4)

    #trainLoader,testLoader=loadCIFAR10(batchsize)
    f=open(filename, 'w')
    for i in range(curEpoch,curEpoch+epoch):
        model.train()
        sum_loss=0

        for batch_idx,(x,target) in enumerate(trainLoader):
            optimizier.zero_grad()
            if adjustLR:
                adjust_lr(optimizier,epoch)
            if useCuda:
                x,target=x.cuda(),target.cuda()
            x,target=Variable(x),Variable(target)
            out=model(x)

            loss=ceritation(out,target)
            sum_loss+=loss.item()
            loss.backward()
            optimizier.step()

            step+=1

            if (batch_idx+1) %checkPoint==0 or (batch_idx+1)==len(trainLoader):
                print('==>>>epoch:{}, batch index:{},step:{}, target loss:{:.6f}'.format(i,batch_idx+1,step,sum_loss/(batch_idx+1)))
                f.write('training ==>>>epoch:{}, batch index:{},step:{}, target loss:{:.6f}'.format(i,batch_idx+1,step,sum_loss/(batch_idx+1)))
                f.write('\n')

            train_loss.append(sum_loss/(batch_idx+1))

            #save model every savepoint steps
            if(step+1)%savePoint==0:
                saveModel(model,i,best_acc,modelPath)
                print('.....save finish......')



        acc=test(net,useCuda=True)

        if earlyStop:
            if acc<best_acc:
                tolearnce_cnt+=1
            else:
                best_acc=acc
                tolearnce_cnt=0
            if tolearnce_cnt>=tolearnce:
                print('early stopping training..........')
                break
        #save model
        if best_acc<acc:
            saveModel(net,epoch,best_acc,modelPath)
            best_acc=acc
    traincsv = pd.DataFrame(train_loss)
    traincsv.to_csv(train_loss_path, index=False)

def saveModel(net,epoch,best_acc,model_path):
    print('saving.....')
    state={
        'net':net.state_dict(),
        'acc':best_acc,
        'epoch':epoch,
    }
    torch.save(state,model_path)

def loadModel(modelPath,net):
    print('==>Resuming from checkpoint...')
    checkpoint=torch.load(modelPath)
    net.load_state_dict(checkpoint['net'])
    best_acc=checkpoint['acc']
    start_epoch=checkpoint['epoch']
    return net,best_acc,start_epoch

def test(model,useCuda=True):
    correct_cnt,sum_loss=0,0
    total_cnt=0

    model.eval()

    for batch_index,(x,target) in enumerate(testLoader):
        x, target = Variable(x,volatile=True), Variable(target,volatile=True)
        if useCuda:
            x, target = x.cuda(), target.cuda()
        out=model(x)
        _,pre_label=torch.max(out.data,1)
        total_cnt+=x.data.size()[0]
        correct_cnt+=(pre_label==target.data).sum()
        correct_cnt=correct_cnt.item()
    acc=(correct_cnt*1.0/float(total_cnt))
    print("acc: ",acc)
    return acc


if __name__ == '__main__':
    # Model
    test_loss_path = './save/test_loss.csv'
    train_loss_path = './save/train_loss.csv'
    #model_path = "/disk/yaoyang/model/new_model.pkl"
    save_path_root="./checkpoint/"
    filename='./save/new_check.txt'
    #model_path = "./save/new_model_cifar100_new.pkl"
    batchsize = 128
    use_cuda = torch.cuda.is_available()

    model_path=save_path_root+args.save

    print('model_save_path is:',model_path)
    trainLoader,testLoader=load_data(batchSize=batchsize)

    if args.data == 'cifar10':
        # net=NewModel1(num_classes=10)
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        #net = MobileNetV2(num_classes=10)
        # net=ResNet50(num_classes=100)
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net=ResNet34(num_classes=10)
        net = NewModel24(num_classes=10)
    elif args.data == 'cifar100':
        # net=NewModel1(num_classes=100)
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        #net = MobileNetV2(num_classes=100)
        # net=ResNet50(num_classes=100)
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ResNet34(num_classes=10)
        net = NewModel24(num_classes=100)


    if args.resume:
        print('==> loading model..')
        net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
        net,best_acc,curEpoch=loadModel(model_path,net)
        #net.load_state_dict(torch.load(model_path))
    else:
        best_acc=0
        curEpoch=0
        print('==> Building model..')
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            #net.apply(weight_init)
    print('current epoch: ', curEpoch)
    print('current best acc: ', best_acc)
    use_cuda=torch.cuda.is_available()
    if args.test:
        test(net,useCuda=use_cuda,)
    else:
        train(net,epoch=120,checkPoint=10,savePoint=500,modelPath=model_path,
          useCuda=use_cuda,best_acc=best_acc,adjustLR=True,curEpoch=curEpoch,earlyStop=True)

