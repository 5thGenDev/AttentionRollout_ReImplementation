import torch
import numpy as np
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
    def __init__(self,model: nn.Module, module='attn.attn_drop') -> None:
        self.model = model
        self.module = module
        self.model = model
        self.outputs = []

    def register_hook(self):
        for name, m in self.model.named_modules():
            if name.endswith(self.module):
                ''' model._modules.get(name) won't work if name is a nested module
                It needs be: model._modules.get(child submodule)._modules.get(grandchild sub)

                yield retains the return value so next time when this is called,
                it won't return the same thing
                '''
                SubModules_list = name.split('.')
                current_module = self.model
                for submod in SubModules_list:
                    if submod.isdigit():  # If it's a number (indicating a layer in nn.Sequential for instance)
                        current_module = current_module[int(submod)]
                    else:
                        current_module = current_module._modules.get(submod)
                yield current_module.register_forward_hook(self._hook)


    def _hook(self, m, input, output):
        self.outputs.append(output)

    def __call__(self, input_tensor):
        self.outputs = []
        self.hook_handlers = list(self.register_hook())

        with torch.no_grad():
            output = self.model(input_tensor)

        mask = rollout(self.outputs)

        for h in self.hook_handlers:
            h.remove()

        return mask
