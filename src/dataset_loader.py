# Copyright (c) EEEM071, University of Surrey

import os
import sys
from torchvision import datasets as D


def build_dataset(args, is_train, trnsfrm=None, training_mode='finetune'):
    if args.data_set == 'Pets':
        folder = 'OxfordIIIPet'
        split = 'trainval' if is_train else 'test'
        root = os.path.join(args.data_location, folder)
        dataset = D.Flowers102(root=root, split=split, transform=trnsfrm, download=True,)
        nb_classes = 37

    if agrs.data_set == 'Flowers':
        folder = 'Oxford102Flowers'
        split = 'train' if is_train else 'test'
        root = os.path.join(args.data_location, folder)
        dataset = D.Flowers102(root=root, split=split, transform=trnsfrm, download=True,)
        nb_classes = 102
    
    else:
        print('dataloader of {} is not implemented .. please add the dataloader under datasets folder.'.format(args.data_set))
        sys.exit(1)
           
    return dataset, nb_classes

