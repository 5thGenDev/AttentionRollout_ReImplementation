# Copyright (c) EEEM071, University of Surrey

import os
import sys
import torch
from torchvision import datasets as D


def build_dataset(args, is_train, trnsfrm=None):
    if args.data_set == 'Pets':
        folder = 'OxfordIIIPet'
        split = 'trainval' if is_train else 'test'
        root = os.path.join(args.data_location, folder)
        dataset = D.Flowers102(root=root, split=split, transform=trnsfrm, download=True,)
        classes = 37

    if agrs.data_set == 'Flowers':
        folder = 'Oxford102Flowers'
        split = 'train' if is_train else 'test'
        root = os.path.join(args.data_location, folder)
        dataset = D.Flowers102(root=root, split=split, transform=trnsfrm, download=True,)
        classes = 102

    if args.data_Set == 'CIFAR10':
        folder = 'CIFAR10'
        root = os.path.join(args.data_location, folder)
        dataset = D.CIFAR10(root=root, transform=trnsfrm, download=True,)
        classes = 10
        
    else:
        print('dataloader of {} is not implemented .. please add the dataloader under datasets folder.'.format(args.data_set))
        sys.exit(1)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"Data loaded: there are {len(dataset)} images.")
    
    return data_loader, classes

