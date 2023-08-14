import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

from attention_visualizer.heatmap import show_mask_on_image
from attention_visualizer.hook_rollout import Hook

if __name__ == '__main__':
    set_random_seed(args.seed)
    args = get_args()
   
