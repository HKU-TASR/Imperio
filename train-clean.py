from core.dataset import FashionMNIST, CIFAR10, TinyImageNet
from models.preact_resnet import PreActResNet18
from models.resnet import ResNet18TinyImagenet
from models.cnn import CNNFashionMNIST
from core.routine import train_clean, save_clean
from core.utils import sync_seed
from torch import nn, optim
import argparse
import logging
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='fmnist', choices=('fmnist', 'cifar10', 'timagenet'))
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps')
parser.add_argument('--seed', type=int, default=2024)
args = parser.parse_args()
sync_seed(args.seed)

########################################################################################################################
out_dir = os.path.join('./checkpoints', '%s_clean' % args.dataset)
assert not os.path.exists(out_dir), 'Looks like you have already run this script'
os.makedirs(out_dir)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(os.path.join(out_dir, "progress.log")), logging.StreamHandler()])

########################################################################################################################
if args.dataset == 'fmnist':
    dataset = FashionMNIST(batch_size=args.batch_size)
    model = CNNFashionMNIST().to(args.device)
    n_epochs, lr_step = 100, 20
elif args.dataset == 'cifar10':
    dataset = CIFAR10(batch_size=args.batch_size)
    model = PreActResNet18().to(args.device)
    n_epochs, lr_step = 500, 100
elif args.dataset == 'timagenet':
    dataset = TinyImageNet(batch_size=args.batch_size)
    model = ResNet18TinyImagenet().to(args.device)
    n_epochs, lr_step = 500, 100

########################################################################################################################
optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.90)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.10)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, n_epochs + 1):
    train_clean(model, dataset.transform, dataset.train_dataloader, optimizer, criterion, epoch, args.device)
    scheduler.step()
save_clean(out_dir, n_epochs, model)
