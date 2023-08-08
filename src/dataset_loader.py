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

# now load a train set and a validation set
def load_dataset(
    dataset_name, 
    root #specify where you download your dataset
):
    match dataset_name:
        case "CIFAR10":
            train_data = datasets.CIFAR10(root=root,train=True,download=True, )
            valid_data = datasets.CIFAR10(root=root,train=False,download=False, )

        case "Flowers102":
            train_data = datasets.Flowers102(root=root,train=True,download=True, )
            valid_data = datasets.Flowers102(root=root,train=False,download=True, )
            
        case "OxfordIIITPet":
            data = datasets.OxfordIIITPet(root=root,download=True, )
            train_data = datasets.CIFAR10(root=root,train=True,download=True, )
            valid_data = datasets.CIFAR10(root=root,train=False,download=True, )

    for sample_idx in range (0, len(training_data)):
        img, label = data[sample_idx] 
        img = Fuse_RGB_Gray_SKetch(img) 
        
    train_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=True,num_workers=8,pin_memory=True)
    
    data = datasets.OxfordIIITPet(root=root,train=False,download=True)
    val_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=False,num_workers=4)
