import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
import time
from models import resnet, lenet

best_accuracy = 0
avg_loss = 0

# Training
def train(net, trainloader, optimizer, criterion, device):
    global best_accuracy
    global avg_loss
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
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        end_t = time.monotonic()

        train_time +=(end_t - start_t)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    et = time.monotonic()
    total_time +=(et - st)
    accuracy = 100.*correct/total
    loss = train_loss/len(trainloader)
    avg_loss += loss

    if accuracy > best_accuracy:
        best_accuracy = accuracy
    
    print('[INFO] Epoch Training Loss: %.3f' % loss)
    print('[INFO] Epoch Training Accuracy: %.3f' % accuracy)
    print('[INFO] Epoch Data Loading Time: %.3f s' % data_time)
    print('[INFO] Epoch Training Time: %.3f s' % train_time)
    
    return total_time, data_time, train_time


# Testing
def test(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


def main(args):
    print()
    total_time = 0
    data_time = 0
    train_time = 0
    device = args.device
    epochs = args.epochs

    # Device details
    if device == 'cuda':
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            num_devices = torch.cuda.device_count()
            device = 'cuda'
            print('[INFO] Using GPU...')
            print(f'[INFO] Device Name: {device_name}')
            print(f'[INFO] Number of devices: {num_devices}')
        else:
            device = 'cpu'
            print('[INFO] Using CPU...')
    
    # Data
    print('\n==> Preparing data...')

    if args.data == 'mnist':
        trainset = torchvision.datasets.MNIST(root='.data', train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root='.data', train=False, download=True, transform=transforms.ToTensor())
        dataset = 'MNIST'
    elif args.data == 'fashion':
        trainset = torchvision.datasets.FashionMNIST(root='.data', train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.FashionMNIST(root='.data', train=False, download=True, transform=transforms.ToTensor())
        datastet = 'FashionMNIST'

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    
    print(f'[INFO] Using Dataset: {dataset}')
    print(f'[INFO] Batch Size: {args.batch_size}')

    # Model
    print('\n==> Building model...')
    
    if args.model == 'resnet':
        net = resnet.ResNet18()
        model = 'ResNet-18'
    elif args.model == 'lenet':
        net = lenet.LeNet()
        model = 'LeNet'

    print(f'[INFO] Training {model} model')
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    print('\n==> Evaluating model...')
    for epoch in range(1, epochs+1):
        print(f'\n[INFO] Epoch: {epoch}')
        total, data, train_t = train(net, trainloader, optimizer, criterion, device)
        total_time += total
        data_time += data
        train_time += train_t
        test(net, testloader, criterion, device)
        scheduler.step()
        print('[UPDATE] Average Training Loss: %.3f' % (avg_loss/epoch))
        
    total_train_time = data_time + train_time
    
    print('\n[INFO] Best Training Accuracy: %.3f' % best_accuracy)
    print('[INFO] Total Data Loading Time: %.3f s' % data_time)
    print('[INFO] Total Epoch Training Time: %.3f s' % train_time)
    print('[INFO] Total Training Time: %.3f s' % total_train_time)
    print('[INFO] Total Training Loop Time: %.3f s' % total_time)
    print()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Benchmarking')
    parser.add_argument('--device', default='cuda', help='GPU or CPU')
    parser.add_argument('--data', default='mnist', help='mnist, fashion')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--model', default='resnet', type=str, help='ResNet-18 or LeNet model')
    parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=10, type=int, help='Epochs to train')
    args = parser.parse_args()
    main(args)
