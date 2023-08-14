import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
from torch import nn


def rollout(encoderblock_outputs):
    result = torch.eye(encoderblock_outputs[0].size(-1))
    
    with torch.no_grad():
        # Taking the mean of the maximum value across all heads
        for multihead in encoderblock_outputs:
            # Max across each multihead
            attention_heads_fused = multihead.max(axis=1)[0]
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2

            # Mean of max values
            a = a / a.sum(dim=-1, keepdim=True)
            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    


class Hook:
    def __init__(self,model: nn.Module, module='attn_drop') -> None:
        """set a hook to get the intermedia results

        Args:
            model (nn.Module): ViTs
            module (str, optional): Name of module. 'attn_drop' for attn matrix; 'block' for outputs of blocks.
        """
        self.model = model
        self.module = module

        # PyTorch does its thing
        self.register_hook()
                
    def register_hook(self):
        for name, m in self.model.named_modules():
            if name.endswith(self.module):
                yield m.register_forward_hook(self._hook)
    
    def _hook(self, m, input, output):
        self.outputs.append(output)
        
    def __call__(self, input_tensor):
        self.outputs = []
        mask = rollout(self.outputs)
        with torch.no_grad():
            output = self.model(input_tensor)        
        
        return mask
