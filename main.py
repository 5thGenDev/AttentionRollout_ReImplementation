import argparse
import sys
import torch

from Architectures import vit 
from src.load_dataset import build_dataset
from src.transform import build_transform

if __name__ == '__main__':
    set_random_seed(args.seed)
    args = get_args()

    transform = build_transform(is_train=True)
    data_loader, classes = build_dataset(args, is_train=True, trnsfrm=transform)
    model = vits.__dict__[args.model](img_size=224, num_classes=classes)
   
