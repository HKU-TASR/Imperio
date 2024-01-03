import argparse
import random
import shutil
import tqdm
import os


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True,
                    help='Path to the unzipped folder downloaded from http://cs231n.stanford.edu/tiny-imagenet-200.zip')
args = parser.parse_args()
train_root = os.path.join(args.path, 'train')

assert os.path.exists(train_root), 'Make sure you have provided the correct path to the unzipped folder'

dst_root = './data/TinyImageNet_200'
with open(os.path.join(args.path, 'words.txt'), 'r') as f:
    classes = dict([line.strip().split('\t') for line in f.readlines()])

for cls_id in tqdm.tqdm(os.listdir(train_root), total=len(os.listdir(train_root)), desc='Copying'):
    if cls_id[0] == '.':
        continue

    class_name = classes[cls_id].lower()
    fnames = os.listdir(os.path.join(train_root, cls_id, 'images'))
    random.shuffle(fnames)
    split = int(len(fnames) * 0.80)
    fnames_train = fnames[:split]
    fnames_test = fnames[split:]

    os.makedirs(os.path.join(dst_root, 'train', class_name))
    for fname in fnames_train:
        src_path = os.path.join(train_root, cls_id, 'images', fname)
        dst_path = os.path.join(dst_root, 'train', class_name, fname.lower())
        shutil.copy(src_path, dst_path)

    os.makedirs(os.path.join(dst_root, 'test', class_name))
    for fname in fnames_test:
        src_path = os.path.join(train_root, cls_id, 'images', fname)
        dst_path = os.path.join(dst_root, 'test', class_name, fname.lower())
        shutil.copy(src_path, dst_path)

print('Done. The TinyImageNet dataset is copied to `%s`.' % dst_root)
