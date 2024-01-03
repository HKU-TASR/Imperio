from core.dataset import FashionMNIST, CIFAR10, TinyImageNet
from models.preact_resnet import PreActResNet18
from models.resnet import ResNet18TinyImagenet
from models.cnn import CNNFashionMNIST
from core.instruction import Instruction
from core.trigger import Trigger
from core.routine import test_clean, test
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
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps')
parser.add_argument('--seed', type=int, default=2024)
args = parser.parse_args()
sync_seed(args.seed)

########################################################################################################################
llm_cache_path = os.path.join('./checkpoints', '%s_%s.pt' % (args.dataset, args.llm.split('/')[1]))
state_dict_path_clean = os.path.join('./checkpoints/%s_clean/epoch-%04d.pt' % (
    args.dataset, [500, 100]['fmnist' in args.dataset]))
state_dict_path_dirty = os.path.join('./checkpoints/%s_backdoor-%s/epoch-%04d.pt' % (
    args.dataset, args.llm.split('/')[1], [500, 100]['fmnist' in args.dataset]))

errors = []
if not os.path.exists(state_dict_path_clean):
    errors.append('python train-clean.py --dataset %s' % args.dataset)
if not os.path.exists(state_dict_path_dirty):
    errors.append('python train-backdoor.py --dataset %s --hf-token %s' % (args.dataset, args.hf_token))
assert len(errors) == 0, 'Run the following scripts first: %s' % str(errors)

########################################################################################################################
if args.dataset == 'fmnist':
    dataset = FashionMNIST(batch_size=args.batch_size)
    model = CNNFashionMNIST()
elif args.dataset == 'cifar10':
    dataset = CIFAR10(batch_size=args.batch_size)
    model = PreActResNet18()
elif args.dataset == 'timagenet':
    dataset = TinyImageNet(batch_size=args.batch_size)
    model = ResNet18TinyImagenet()

########################################################################################################################
# Clean model
########################################################################################################################
state_dict = torch.load(state_dict_path_clean, map_location=args.device)
model.load_state_dict(state_dict['model'], strict=True)
model = model.to(args.device).eval()

ACC_baseline = test_clean(model, dataset.transform, dataset.defense_test_dataloader, args.device)
print('===============================================')
print('Baseline ACC: %.2f%%' % ACC_baseline)
del state_dict
########################################################################################################################
# Victim model
########################################################################################################################
instruction = Instruction(synonyms=dataset.synonyms)
trigger = Trigger(args.llm, args.epsilon, dataset.input_shape, args.hf_token, device=args.device, load_llm=False)
trigger.build_cache(instruction, llm_cache_path)

state_dict = torch.load(state_dict_path_dirty, map_location=args.device)
model.load_state_dict(state_dict['model'], strict=True)
model = model.to(args.device).eval()
trigger.load_state_dict(state_dict['trigger'], strict=True)
trigger = trigger.to(args.device).eval()

ACC = test_clean(model, dataset.transform, dataset.defense_test_dataloader, args.device)
print('Victim ACC: %.2f%% (%+.2f%%)' % (ACC, ACC - ACC_baseline))
print('===============================================')
print('Per-class ASR')
ASRs_known, ASRs_unknown = [], []
for target_id in range(dataset.n_classes):
    ASR_known = test(model, trigger, dataset.transform, instruction, dataset.defense_test_dataloader, args.device,
                     backdoor=target_id, unknown=False)
    ASR_unknown = test(model, trigger, dataset.transform, instruction, dataset.defense_test_dataloader, args.device,
                       backdoor=target_id, unknown=True)
    print('[%s] Known: %.2f%% | Unknown: %.2f%%' % (dataset.idx_to_class[target_id], ASR_known, ASR_unknown))
    ASRs_known.append(ASR_known)
    ASRs_unknown.append(ASR_unknown)
print('-----------------------------------------------')
print('ASR (Known)  : %.2f%%' % float(np.mean(ASRs_known)))
print('ASR (Unknown): %.2f%%' % float(np.mean(ASRs_unknown)))
