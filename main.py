import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


from Architectures import vit 
from src.load_dataset import build_dataset
from src.transform import build_transform
from src.optimizer import init_optimizer
from src.schedulers import init_lr_scheduler
from args import argument_parser, optimizer_kwargs, lr_scheduler_kwargs

parser = argument_parser()
args = parser.parse_args()


def save_checkpoint(model, optimizer, epoch, directory, filename='checkpoint.pth'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = os.path.join(directory, filename)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    

def train(model, train_loader, optimizer, criterion, num_epochs, device, save_dir='checkpoints', save_name='vit_checkpoint.pth'):
    model.to(device)

    train_losses = []
    best_train_loss = float('inf') 

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            save_checkpoint(model, optimizer, epoch, directory=save_dir, filename=save_name)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

### To counter "name args is assigned before global declaration, put everything to def main()
def main():
    global args

    ### CUDA stuffs
    if not args.use_avai_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    if args.use_cpu:
        device = 'cpu'
    cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ### main pipeline
    transform = build_transform(is_train=True)
    train_loader, classes = build_dataset(args, is_train=True, trnsfrm=transform)
    model = vit.__dict__[args.model](img_size=224, num_classes=classes)
    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))
    criterion = nn.CrossEntropyLoss()
    train(model, train_loader, optimizer, criterion, num_epochs=args.epochs, device=device, save_dir=args.save_dir, save_name=args.model)
    
if __name__ == '__main__':
    main()
