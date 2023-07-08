# Download the dataset
import torch
from torchvision import datasets, transforms

transforms_chain = transforms.Compose([
  transforms.Resize(size=256, interpolation=3),
  transforms.CenterCrop(size=224),
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Don't visualize img after applying normalize
])

data = datasets.CIFAR10(
    "data",
    transform=transforms_chain,
    download=True,
)

# For visualize some img from dataset
'''
from torchvision import utils
import torchvision.transforms.functional as tfF
imgs = torch.stack([data[i][0] for i in range(8)])
tfF.to_pil_image(utils.make_grid(imgs,))
'''
