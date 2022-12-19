'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchvision import datasets

import argparse
import time
from torchinfo import summary
from resnet import resnet

best_acc = 0  # best test accuracy

# Training
def train(net, trainloader, optimizer, criterion, device):
    global best_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    data_time = 0
    train_time = 0
    total_time = 0

    st = time.monotonic()
    for inputs, targets in trainloader:
        start=time.monotonic()
        inputs, targets = inputs.to(device), targets.to(device)
        end = time.monotonic()

        data_time += end - start
        start_t = time.monotonic()
        optimizer.zero_grad()
        outputs,x = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        end_t = time.monotonic()

        train_time +=(end_t - start_t)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        if acc > best_acc:
            best_acc = acc

    et = time.monotonic()
    total_time +=(et - st)
    loss = train_loss/len(trainloader)

    print('Training loss: %.3f' % loss)
    print('Epoch Training Time: %.3f s' % train_time)
    print('Epoch Data Loading Time: %.3f s' % data_time)
    print('Top accuracy: %.3f' % best_acc)
    print()

    return total_time


# Testing
def test(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,x = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


def main(args):
    total_time = 0
    device = args.device

    if device == 'cuda':
        if torch.cuda.is_available():
            print('Using GPU...')
            device = 'cuda'
        else:
            print('Using CPU...')
            device = 'cpu'

    # Data
    print('\n==> Preparing data...')
    train_dataset = datasets.MNIST(root='data', 
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = datasets.MNIST(root='data', 
                                train=False, 
                                transform=transforms.ToTensor())


    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=128, 
                            shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=128, 
                            shuffle=False)
      # Model
    print('\n==> Building model...')
    net = resnet.resnet18(10)
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'nesterov':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, nesterov=True, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        raise ValueError('Optimizer not supported')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(5):
        print(f'\nEpoch: {epoch+1}')
        total_time += train(net, train_loader, optimizer, criterion, device)
        test(net, test_loader, criterion, device)
        scheduler.step()

    print('\nTotal Training Time: %.3f s' % total_time)
    print('\nArchitecture Summary:')
    #summary(net, (128, 3, 32, 32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--data_path', default='./data', type=str, help='Path to data')
    parser.add_argument('--device', default='cuda', help='GPU or CPU')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers')
    parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer')
    args = parser.parse_args()
    main(args)