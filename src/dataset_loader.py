# Copyright (c) EEEM071, University of Surrey

import os
import sys

from datasets.Pets import pets

def build_dataset(args, is_train, trnsfrm=None, training_mode='finetune'):
    if args.data_set == 'Pets':
        split = 'trainval' if is_train else 'test'
        dataset = pets(os.path.join(args.data_location, 'Pets_dataset'), split=split, transform=trnsfrm)
        nb_classes = 37

    if agrs.data_set == 'Flowers':
        split = 'train' if is_train else 'test'
        dataset = flowers(os.path.join(args.data_location, 'Flowers_dataset'), split=split, transform=trnsfrm)
        nb_classes = 102
    
    else:
        print('dataloader of {} is not implemented .. please add the dataloader under datasets folder.'.format(args.data_set))
        sys.exit(1)
           
    return dataset, nb_classes

