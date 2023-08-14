import argparse
import sys
import torch

from Architectures import vit 

if __name__ == '__main__':
    set_random_seed(args.seed)
    args = get_args()

    
    model = vits.__dict__[args.model](img_size=224, num_classes=classes)
   
