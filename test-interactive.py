from core.dataset import FashionMNIST, CIFAR10, TinyImageNet
from models.preact_resnet import PreActResNet18
from models.resnet import ResNet18TinyImagenet
from models.cnn import CNNFashionMNIST
from matplotlib import pyplot as plt
from core.instruction import Instruction
from core.trigger import Trigger
from core.utils import sync_seed
import numpy as np
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=('fmnist', 'cifar10', 'timagenet'))
parser.add_argument('--hf-token', type=str, required=True, help='HuggingFace token')
parser.add_argument('--llm', type=str, default='meta-llama/Llama-2-13b-chat-hf')
parser.add_argument('--epsilon', type=float, default=0.05, help='Maximum change to the clean image')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps')
parser.add_argument('--seed', type=int, default=2024)
args = parser.parse_args()
sync_seed(args.seed)

########################################################################################################################
llm_cache_path = os.path.join('./checkpoints', '%s_%s.pt' % (args.dataset, args.llm.split('/')[1]))
state_dict_path_dirty = os.path.join('./checkpoints/%s_backdoor-%s/epoch-%04d.pt' % (
    args.dataset, args.llm.split('/')[1], [500, 100]['fmnist' in args.dataset]))

errors = []
if not os.path.exists(state_dict_path_dirty):
    errors.append('python train-backdoor.py --dataset %s --hf-token %s' % (args.dataset, args.hf_token))
assert len(errors) == 0, 'Run the following scripts first: %s' % str(errors)

########################################################################################################################
if args.dataset == 'fmnist':
    dataset = FashionMNIST(batch_size=1)
    model = CNNFashionMNIST()
elif args.dataset == 'cifar10':
    dataset = CIFAR10(batch_size=1)
    model = PreActResNet18()
elif args.dataset == 'timagenet':
    dataset = TinyImageNet(batch_size=1)
    model = ResNet18TinyImagenet()

########################################################################################################################
# Victim model
########################################################################################################################
state_dict = torch.load(state_dict_path_dirty, map_location=args.device)
model.load_state_dict(state_dict['model'], strict=True)
model = model.to(args.device).eval()

instruction = Instruction(synonyms=dataset.synonyms)
trigger = Trigger(args.llm, args.epsilon, dataset.input_shape, args.hf_token, device=args.device)
trigger.load_state_dict(state_dict['trigger'], strict=False)
trigger = trigger.to(args.device).eval()

########################################################################################################################
print('Supported Classes: %s' % instruction.labels)
with torch.no_grad():
    for data_clean, target in dataset.defense_test_dataloader:
        data_clean = data_clean.to(args.device)
        target = target.to(args.device)
        if not torch.equal(torch.argmax(model(dataset.transform(data_clean)), dim=1), target):
            continue

        img_clean = data_clean.detach().cpu()[0].permute(1, 2, 0).numpy()
        cls_clean = target[0].item()

        prompt = input('\nTrue Class: %s\nInstruction: ' % (dataset.idx_to_class[cls_clean]))
        mask = trigger(prompts=[instruction.apply_template(prompt)]).to(args.device)
        data_dirty = torch.clamp(data_clean + mask, min=0.0, max=1.0)
        pred_dirty = model(dataset.transform(data_dirty))

        img_mask = mask.detach().cpu()[0].permute(1, 2, 0).numpy()
        img_mask = (img_mask - np.min(img_mask)) / (np.max(img_mask) - np.min(img_mask))
        img_dirty = data_dirty.detach().cpu()[0].permute(1, 2, 0).numpy()
        cls_dirty = torch.argmax(pred_dirty, dim=1)[0].item()

        print('Generating visualization...')
        plt.figure(dpi=300, figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.title('True: ' + dataset.idx_to_class[cls_clean])
        plt.imshow(img_clean, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title('Trigger')
        plt.imshow(img_mask, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.title('Predicted: ' + dataset.idx_to_class[cls_dirty])
        plt.imshow(img_dirty, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
