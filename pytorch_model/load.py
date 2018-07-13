import torchvision
import torch
import torchvision.transforms as transforms
import torch.utils.data
from PascalLoader import DataLoader

def loadCIFAR10(batchSize):
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='/disk/yaoyang/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='/disk/yaoyang/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader

def loadCIFAR100(batchSize):
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR100(root='/disk/yaoyang/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR100(root='/disk/yaoyang/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=1)

    return trainloader, testloader

def load_PASCAL(batchsize):
    pascal_path='/disk/yaoyang/data/VOCdevkit/VOC2012'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        # transforms.Scale(256),
        # transforms.CenterCrop(227),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_data = DataLoader(pascal_path, 'trainval', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batchsize,
                                               shuffle=True,
                                               num_workers=1)

    val_data = DataLoader(pascal_path, 'val', transform=val_transform, random_crops=10)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=batchsize,
                                             shuffle=False,
                                             num_workers=1)
    N = len(train_data.names)
    iter_per_epoch = int(N / batchsize)
    return train_loader, val_loader
