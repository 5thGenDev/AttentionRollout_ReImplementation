from datasets_preprocessing.simple_pipeline import build_transform, load_dataset
import Architectures.vision_transformer as vit

#  Create Model
device = "cuda"
model = vit_small(patch_size=16,img_size=[224], num_classes=10)
# model = vit_small(patch_size=16,img_size=[224], num_classes=10, num_heads=768)  # For HydraAttention, num_heads=embed_dim=768
model = model.to(device)

# load the pretrained weights
state = torch.load("SiT_Small_ImageNet.pth")
state = {k.replace("module.backbone.",""):v for k,v in state['teacher'].items()} # extract the model weights!
print("Loading weights: ", model.load_state_dict(state,strict=False))

# Load the pretrained weights
from torch import optim
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)

import tqdm
def train_one_epoch(dl,model, opt, device):
    model.train()
    pbar = tqdm.tqdm(dl)
    for x,label in pbar:
        x,label = x.to(device),label.to(device)
        pred = model(x)
        loss = nn.functional.cross_entropy(pred,label)
        loss.backward()
        opt.step()
        opt.zero_grad()
        pbar.set_description(f"loss: {loss.item():.3f}")

@torch.no_grad()
def eval_one_epoch(dl,model, device):
    model.eval()
    pbar = tqdm.tqdm(dl)
    def _f():
        for x,label in pbar:
            x,label = x.to(device),label.to(device)
            pred = model(x)
            loss = nn.functional.cross_entropy(pred,label)
            acc = (pred.argmax(1)==label).float().mean()
            pbar.set_description(f"loss: {loss.item():.3f}, acc: {acc.item():.3f}")
            yield pred.argmax(1),label
    out = list(_f())
    pred, y = list(zip(*out))
    return torch.cat(pred),torch.cat(y)


# this implementation achieves roughly an accuracy of 0.64.
num_epoch = 30
log=[]
for epoch in range(num_epoch):
    train_one_epoch(train_dl,model, optimizer, device)
    if epoch%2==0:
        pred,y = eval_one_epoch(val_dl,model, device)
        acc = (pred==y).float().mean().item()
        log.append(acc)

        print(epoch, acc)
