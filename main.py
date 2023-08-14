import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


from Architectures import vit 
from src.load_dataset import build_dataset
from src.transform import build_transform
from src.optimizer import employ_optimizer
from src.schedulers import init_lr_scheduler
from args import argument_parser, optimizer_kwargs, lr_scheduler_kwargs


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


if __name__ == '__main__':
    set_random_seed(args.seed)
    
    parser = argument_parser()
    args = parser.parse_args()

    ### CUDA stuffs
    if not args.use_avai_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    if use_gpu:
        print(f"Currently using GPU {args.gpu_devices}")
        cudnn.benchmark = True
    else:
        warnings.warn("Currently using CPU, however, GPU is highly recommended")

    ### main pipeline
    transform = build_transform(is_train=True)
    train_loader, classes = build_dataset(args, is_train=True, trnsfrm=transform)
    model = vit.__dict__[args.model](img_size=224, num_classes=classes)
    optimizer = employ_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))
    criterion = nn.CrossEntropyLoss()
    train(model, train_loader, optimizer, criterion, num_epochs=args.epochs, device=device)
   
