# Copyright (c) EEEM071, University of Surrey

import os.path as osp

from PIL import Image
from torchvision import datasets
from src.multimodal_transform import Fuse_RGB_Gray_Sketch

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
    root="data"
):
    match dataset_name:
        case "CIFAR10":
            data = datasets.CIFAR10(root="data",download=True, )
            train_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=True,num_workers=8,pin_memory=True)
            
            data = datasets.CIFAR10(root="data",False,download=True)
            val_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=False,num_workers=4)

        case "Flowers102":
            data = datasets.Flowers102(root="data",download=True, )
            train_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=True,num_workers=8,pin_memory=True)
            
            data = datasets.Flowers102(root="data",False,download=True)
            val_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=False,num_workers=4)

        case "OxfordIIITPet":
            data = datasets.OxfordIIITPet(root="data",download=True, )
            train_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=True,num_workers=8,pin_memory=True)
            
            data = datasets.OxfordIIITPet(root="data",False,download=True)
            val_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=False,num_workers=4)
