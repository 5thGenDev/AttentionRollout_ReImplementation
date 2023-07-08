# Download the dataset
import torch
from torchvision import datasets, transforms as tf, utils
import torchvision.transforms.functional as tfF
tfms = tf.Compose([
  tf.Resize(size=256, interpolation=3),
  # tf.Resize(size=256),
  tf.CenterCrop(size=224),
  tf.ToTensor(),
  # tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
data = datasets.CIFAR10(
    "data",
    transform=tfms,
    download=True,
)
imgs = torch.stack([data[i][0] for i in range(8)])
tfF.to_pil_image(utils.make_grid(imgs,))
