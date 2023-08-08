# Copyright (c) EEEM071, University of Surrey

import os.path as osp

from PIL import Image
from torchvision import datasets
from src.multimodal_transform import Fuse_RGB_Gray_Sketch

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise OSError(f"{img_path} does not exist")
    while not got_img:
        try:
            # Image.open(img_path) return PIL image
            img = Fuse_RGB_Gray_Sketch(Image.open(img_path))
            got_img = True
        except OSError:
            print(
                f'IOError incurred when reading "{img_path}". Will redo. Don\'t worry. Just chill.'
            )
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        return img, pid, camid, img_path



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
