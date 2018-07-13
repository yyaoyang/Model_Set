import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data
import torch
#from Pascal.PascalLoader import DataLoader
from PascalLoader import DataLoader
from models import *
import os, sys, numpy as np
import matplotlib.pyplot as plt
import torch.optim
import argparse
from logging import Logger
from torch.autograd import Variable
CORES=4
from sklearn.metrics import average_precision_score

parser = argparse.ArgumentParser(description='Train network on Pascal VOC 2007')
parser.add_argument('--pascal_path','-path', type=str, help='Path to Pascal VOC 2007 folder')
parser.add_argument('--model', default=None, type=str, help='Pretrained model')
# parser.add_argument('--freeze', dest='evaluate', action='store_true', help='freeze layers up to conv5')
parser.add_argument('--freeze', default=None, type=int, help='freeze layers up to conv5')
parser.add_argument('--fc', default=None, type=int, help='load fc6 and fc7 from model')
parser.add_argument('--gpu', default=None, type=int, help='gpu id')
parser.add_argument('--epochs', default=160, type=int, help='gpu id')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=100, type=int, help='batch size')
parser.add_argument('--checkpoint', default='./save/checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--crops', default=10, type=int, help='number of random crops during testing')
args = parser.parse_args()
# args = parser.parse_args([
#    '/net/hci-storage02/groupfolders/compvis/datasets/VOC2007/',
#    '--gpu','0',
# ])
def adjust_lr(optimizer,epoch):
    lr=args.lr*(0.1**(epoch//30))
    if lr>=1e-6:
        for param_group in optimizer.param_groups:
            param_group['lr']=lr

def compute_mAP(labels,outputs):
    y_true=labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i],y_pred[i]))
    return np.mean(AP)


def load():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomSizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        # transforms.Scale(256),
        # transforms.CenterCrop(227),
        transforms.RandomSizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_data=DataLoader(args.pascal_path,'trainval',transform=train_transform)
    train_loader=torch.utils.data.DataLoader(dataset=train_data,
                                             batch_size=args.batch,
                                             shuffle=True,
                                             num_workers=CORES)
    val_data = DataLoader(args.pascal_path, 'val', transform=val_transform, random_crops=args.crops)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=args.batch,
                                             shuffle=False,
                                             num_workers=CORES)
    N=len(train_data.names)
    iter_per_epoch= int(N/args.batch)

    # net=NewModel1(num_classes=100)
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2(num_classes=100)
    # net=ResNet50(num_classes=100)
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ResNet34(num_classes=10)
    net = NewModel24(num_classes=21)

    if args.gpu is not None:
        net.cuda()
    if args.model is not None:
        net.load(args.model,args.fc)
    if args.freeze is not None:
        for i,(name,param) in enumerate(net.named_parameters()):
            if 'conv' in name or 'features' in name:
                param.requires_grad=False
    criterion=nn.MultiLabelSoftMarginLoss()
    optimizer=torch.optim.SGD(filter(lambda p:p.requires_grad,net.parameters()),
                              lr=args.lr,momentum=0.9,weight_decay=0.0001)
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint+'/train')
        os.makedirs(args.checkpoint + '/test')
    logger_train=Logger(args.checkpoint+'/train')
    logger_test = Logger(args.checkpoint + '/test')

    ############## TRAINING ###############
    print('Start training: lr %f, batch size %d' % (args.lr, args.batch))
    print('Checkpoint: ' + args.checkpoint)
    # Train the Model
    steps = args.iter_start
    for epoch in range(iter_per_epoch * args.iter_start, args.epochs):
        adjust_lr(optimizer,epoch)
        mAP=[]
        for i,(images,labels) in enumerate(train_loader):
            images=Variable(images)
            labels=Variable(labels)
            if args.gpu is not None:
                images=images.cuda()
                labels=labels.cuda()
            optimizer.zero_grad()
            outputs=net(images)
            mAP.append(compute_mAP(labels,outputs))

            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            loss=loss.cpu().data.numpy()

            if steps%100==0:
                print('training----[%d/%d] %d), Loss: %.3f, mAP %.2f%%' % (epoch + 1, args.epochs, steps, loss, 100 * np.mean(mAP[-20:])))

            if steps % 20 == 0:
                logger_train.scalar.summary('mAP', np.mean(mAP[-20:]), steps)
                logger_train.scalar.summary('loss', loss, steps)
                data = images.cpu().data.numpy().transpose([0, 2, 3, 1])
                logger_train.image.summary('input', data[:10], steps)

            steps += 1
        if epoch%5==0:
            torch.save(net.state_dict(), args.checkpoint)
            print("----------save finish----------------")
            # net.save(args.checkpoint, epoch + 1)
            # print('Saved: ' + args.checkpoint)
        # if os.path.exists(args.checkpoint + '/stop.txt'):
        #     # break without using CTRL+C
        #     break
        test(net,criterion,logger_test,val_loader,steps)


def test(net, criterion, logger, val_loader, steps):
    mAP = []
    net.eval()
    for i, (images, labels) in enumerate(val_loader):
        images = images.view((-1, 3, 227, 227))
        images = Variable(images, volatile=True)
        if args.gpu is not None:
            images = images.cuda()

        # Forward + Backward + Optimize
        outputs = net(images)
        outputs = outputs.cpu().data
        outputs = outputs.view((-1, args.crops, 21))
        outputs = outputs.mean(dim=1).view((-1, 21))

        # score = tnt.meter.mAPMeter(outputs, labels)
        mAP.append(compute_mAP(labels, outputs))

    # if logger is not None:
    #     logger.scalar.summary('mAP', np.mean(mAP), steps)
    print('TESTING: %d), mAP %.2f%%' % (steps, 100 * np.mean(mAP)))
    net.train()
if __name__ == '__main__':
    load()



