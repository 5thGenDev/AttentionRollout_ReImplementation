import PIL
from torchvision import datasets, transforms

def build_transform(
    is_train, 
    input_size, 
    mean=(0.485, 0.456, 0.406), 
    std=(0.229, 0.224, 0.225)
):
    # train transform
    if is_train:
      # this should always dispatch to transforms_imagenet_train
      transform = transforms.Compose([
          transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=mean, std=std),
      ])
      return transform
    
    # eval transform
    t = []
    if input_size <= 224:
      crop_pct = 224 / 256
    else:
      crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
      transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))
    
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

# now load a train set and a validation set
def load_dataset(
    dataset_name, 
    root="data"
):
    match dataset_name:
        case "CIFAR10":
            data = datasets.CIFAR10(root,transform=build_transform(True,224),download=True, )
            train_dl = torch.utils.data.DataLoader(data,64,shuffle=True,num_workers=8,pin_memory=True)
            
            data = datasets.CIFAR10(root,False,transform=build_transform(False,224),download=True)
            val_dl = torch.utils.data.DataLoader(data,64,shuffle=False,num_workers=4)

        case "Flowers102":
            data = datasets.Flowers102(root,transform=build_transform(True,224),download=True, )
            train_dl = torch.utils.data.DataLoader(data,64,shuffle=True,num_workers=8,pin_memory=True)
            
            data = datasets.Flowers102(root,False,transform=build_transform(False,224),download=True)
            val_dl = torch.utils.data.DataLoader(data,64,shuffle=False,num_workers=4)

        case "OxfordIIITPet":
            data = datasets.OxfordIIITPet(root,transform=build_transform(True,224),download=True, )
            train_dl = torch.utils.data.DataLoader(data,64,shuffle=True,num_workers=8,pin_memory=True)
            
            data = datasets.OxfordIIITPet(root,False,transform=build_transform(False,224),download=True)
            val_dl = torch.utils.data.DataLoader(data,64,shuffle=False,num_workers=4)
