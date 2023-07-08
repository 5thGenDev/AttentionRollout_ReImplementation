import torch
from torchvision import datasets, transforms

transforms_chain = transforms.Compose([
  transforms.Resize(size=256, interpolation=3),
  transforms.CenterCrop(size=224),
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Don't visualize img after applying normalize
])

data = datasets.Flowers102(
    root="data",
    transform=transforms_chain,
    download=True,
)



train_dl = torch.utils.data.DataLoader(data,64,shuffle=True,num_workers=8,pin_memory=True)

data = datasets.Flowers102("data",False,transform=build_transform(False,224),download=True)
