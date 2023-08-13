import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
from torch import nn


def rollout(outputs, discard_ratio, head_fusion):
    result = torch.eye(outputs[0].size(-1))
    with torch.no_grad():
        for attention in outputs:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
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
    def __init__(self,model: nn.Module, module='attn_drop',head_fusion="mean", discard_ratio=0.9, mode="output") -> None:
        """set a hook to get the intermedia results

        Args:
            model (nn.Module): ViTs
            module (str, optional): Name of module. 'attn_drop' for attn matrix; 'block' for outputs of blocks.
        """
        assert mode in ["input", "output"]
        self.mode = mode 
        self.model = model
        self.module = module
        
    def register_hook(self):
        for name, m in self.model.named_modules():
            if name.endswith(self.module):
                yield m.register_forward_hook(self._hook)
    
    def _hook(self, m, input, output):
        if self.mode == "output":
            self.outputs.append(output)
        else:
            self.outputs.append(input)
        
    def __call__(self, input_tensor):
        self.outputs=[]
        
        # Prior to the line: "for h in hook_handlers: h.remove()"
        # Collects many heads-outputs of Scaled Dot-Product Attention or Hydra Attention
        hook_handlers = list(self.register_hook()) 
        
        # Not tracking gradient to free up memory
        with torch.no_grad():
            output = self.model(input_tensor)        

        mask = rollout(self.outputs, self.discard_ratio, self.head_fusion)
        
        # Remove registered forward and backward hook to free up memory after use
        for h in hook_handlers: h.remove()
        
        return mask
