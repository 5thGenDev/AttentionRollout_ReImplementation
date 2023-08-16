# Copyright (c) EEEM071, University of Surrey

import os
import sys
import torch
from torchvision import datasets as D
from src.pets import OxfordIIIPets
from src.flowers import Flowers102


def build_dataset(args, is_train, trnsfrm=None):
    if args.data_set == 'Pets':
        split = 'trainval' if is_train else 'test'
        root = os.path.join(args.data_location)
        dataset = D.OxfordIIITPet(root=root, split=split, transform=trnsfrm, download=True,)
        classes = 37

    if args.data_set == 'Flowers':
        split = 'train' if is_train else 'test'
        root = os.path.join(args.data_location)
        dataset = D.Flowers102(root=root, split=split, transform=trnsfrm, download=True,)
        classes = 102

    if args.data_set == 'CIFAR10':
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

