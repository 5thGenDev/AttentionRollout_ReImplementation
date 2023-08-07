# Copyright (c) EEEM071, University of Surrey

import os.path as osp

import PIL
from torchvision import datasets
from preprocessing.transforms import build_transforms

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise OSError(f"{img_path} does not exist")
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            got_img = True
        except OSError:
            print(
                f'IOError incurred when reading "{img_path}". Will redo. Don\'t worry. Just chill.'
            )
            pass
    return img

# now load a train set and a validation set
def load_dataset(
    dataset_name, 
    root="data"
):
    match dataset_name:
        case "CIFAR10":
            data = datasets.CIFAR10(root,transform=build_transform(True,224),download=True, )
            train_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=True,num_workers=8,pin_memory=True)
            
            data = datasets.CIFAR10(root,False,transform=build_transform(False,224),download=True)
            val_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=False,num_workers=4)

        case "Flowers102":
            data = datasets.Flowers102(root,transform=build_transform(True,224),download=True, )
            train_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=True,num_workers=8,pin_memory=True)
            
            data = datasets.Flowers102(root,False,transform=build_transform(False,224),download=True)
            val_dl = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=False,num_workers=4)

        case "OxfordIIITPet":
            data = datasets.OxfordIIITPet(root,transform=build_transform(True,224),download=True, )
            train_dl = torch.utils.data.DataLoader(dataset=data,batch_size=64,shuffle=True,num_workers=8,pin_memory=True)
            
            data = datasets.OxfordIIITPet(root,False,transform=build_transform(False,224),download=True)
            val_dl = torch.utils.data.DataLoader(dataset=data,batch_size=64,shuffle=False,num_workers=4)
