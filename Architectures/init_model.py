import torch.utils.model_zoo as model_zoo
import torchvision

model_urls = {
    "vit_b_16": "http://download.pytorch.org/models/vit_b_16-c867db91.pth",
    "vit_l_32": "http://download.pytorch.org/models/vit_l_32-c7638314.pth",
    "vit_h_14": "http://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth",
}

def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    for param_tensor in model_dict:
        print(param_tensor, "\t", model_dict[param_tensor].size())
    model.load_state_dict(model_dict)
    print(f"Initialized model with pretrained weights from {model_url}")
