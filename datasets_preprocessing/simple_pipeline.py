import PIL
from torchvision import datasets

# now load a train set and a validation set
def load_dataset(
    dataset_name, 
    root="data"
):
    match dataset_name:
        case "CIFAR10":
            data = datasets.CIFAR10(root,transform=build_transform(True,224),download=True, )
            train_dl = torch.utils.data.DataLoader(dataset=data,batch_size=64,shuffle=True,num_workers=8,pin_memory=True)
            
            data = datasets.CIFAR10(root,False,transform=build_transform(False,224),download=True)
            val_dl = torch.utils.data.DataLoader(dataset=data,batch_size=64,shuffle=False,num_workers=4)

        case "Flowers102":
            data = datasets.Flowers102(root,transform=build_transform(True,224),download=True, )
            train_dl = torch.utils.data.DataLoader(dataset=data,batch_size=64,shuffle=True,num_workers=8,pin_memory=True)
            
            data = datasets.Flowers102(root,False,transform=build_transform(False,224),download=True)
            val_dl = torch.utils.data.DataLoader(dataset=data,batch_size=64,shuffle=False,num_workers=4)

        case "OxfordIIITPet":
            data = datasets.OxfordIIITPet(root,transform=build_transform(True,224),download=True, )
            train_dl = torch.utils.data.DataLoader(dataset=data,batch_size=64,shuffle=True,num_workers=8,pin_memory=True)
            
            data = datasets.OxfordIIITPet(root,False,transform=build_transform(False,224),download=True)
            val_dl = torch.utils.data.DataLoader(dataset=data,batch_size=64,shuffle=False,num_workers=4)
