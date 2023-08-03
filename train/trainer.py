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


def train(num_epoch):
    log=[]
    for epoch in range(num_epoch):
        train_one_epoch(train_dl,model, optimizer, device)
        if epoch%10==0:
            pred,y = eval_one_epoch(val_dl,model, device)
            acc = (pred==y).float().mean().item()
            log.append(acc)
    
            print(epoch, acc)
  
