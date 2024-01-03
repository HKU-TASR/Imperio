from core.dataset import FashionMNIST, CIFAR10, TinyImageNet
from models.preact_resnet import PreActResNet18
from models.resnet import ResNet18TinyImagenet
from models.cnn import CNNFashionMNIST
from core.instruction import Instruction
from core.trigger import Trigger
from core.routine import train, save
from core.utils import sync_seed
from torch import nn, optim
import itertools
import argparse
import logging
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=('fmnist', 'cifar10', 'timagenet'))
parser.add_argument('--hf-token', type=str, required=True, help='HuggingFace token')
parser.add_argument('--llm', type=str, default='meta-llama/Llama-2-13b-chat-hf')
parser.add_argument('--p', type=float, default=0.10, help='poison ratio (poison p% in a minibatch)')
parser.add_argument('--epsilon', type=float, default=0.05, help='Maximum change to the clean image')
parser.add_argument('--lambda-bi', type=float, default=1.0, help='Backdoor importance')
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps')
parser.add_argument('--seed', type=int, default=2024)
args = parser.parse_args()
sync_seed(args.seed)

########################################################################################################################
llm_cache_path = os.path.join('./checkpoints', '%s_%s.pt' % (args.dataset, args.llm.split('/')[1]))
out_dir = os.path.join('./checkpoints', '%s_backdoor-%s' % (args.dataset, args.llm.split('/')[1]))
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
instruction = Instruction(synonyms=dataset.synonyms)
trigger = Trigger(args.llm, args.epsilon, dataset.input_shape, args.hf_token, device=args.device).to(args.device)
trigger.build_cache(instruction, llm_cache_path)
trigger.free_llm()

########################################################################################################################
# Define optimization variables and begin training
########################################################################################################################
optimizer = optim.SGD(params=itertools.chain(trigger.parameters(), model.parameters()), lr=0.01, momentum=0.90)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.10)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, n_epochs + 1):
    train(model, trigger, dataset.transform, instruction,
          dataset.train_dataloader, optimizer, criterion, args.p, epoch, args.device, lambda_bi=args.lambda_bi)
    scheduler.step()
save(out_dir, n_epochs, trigger, model)