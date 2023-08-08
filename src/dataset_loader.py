# Copyright (c) EEEM071, University of Surrey

import os.path as osp

from PIL import Image
from torchvision import datasets
from src.multimodal_transform import Fuse_RGB_Gray_Sketch, build_test_transform

# img = Fuse_RGB_Gray_SKetch(PIL_image) 
'''
data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
X, label = training_data[sample_idx]
matplotlib.pyplot.imshow(X, cmap="gray") where X = PIL_image
'''

import os
import sys

from src.pets_flowers import pets

def build_dataset(args, is_train, trnsfrm=None, training_mode='finetune'):
    if args.data_set == 'Pets':
        split = 'trainval' if is_train else 'test'
        dataset = pets(os.path.join(args.data_location, 'Pets_dataset'), split=split, transform=trnsfrm)
        
        nb_classes = 37

    else:
        print('dataloader of {} is not implemented .. please add the dataloader under datasets folder.'.format(args.data_set))
        sys.exit(1)
       

        
        
    return dataset, nb_classes



# now load a train set and a validation set
def load_dataset(
    dataset_name, 
    root #specify where you download your dataset
):
    match dataset_name:
        case "CIFAR10":
            train_data = datasets.CIFAR10(root=root,train=True,download=True, )
            valid_data = datasets.CIFAR10(root=root,train=False,download=False, )

    for sample_idx in range (0, len(training_data)):
        img, label = data[sample_idx] 
        img = Fuse_RGB_Gray_SKetch(img) 
        
    train_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=True,num_workers=8,pin_memory=True)
    val_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=False,num_workers=4)
