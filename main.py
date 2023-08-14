import argparse
import sys
import torch
import torch.backends.cudnn as cudnn


from Architectures import vit 
from src.load_dataset import build_dataset
from src.transform import build_transform
from src.optimizer import employ_optimizer
from src.schedulers import init_lr_scheduler


def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, save_path='vit_checkpoint.pth'):
    """
    Train a Vision Transformer model.

    Parameters:
    - model: the ViT model
    - train_loader: data loader for training data
    - val_loader: data loader for validation data
    - optimizer: the optimizer
    - criterion: the loss function
    - num_epochs: number of epochs to train
    - device: 'cuda' or 'cpu'
    - save_path: path to save the best model checkpoint
    """

    # Store the loss and accuracy values for later visualization
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0  

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

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        val_accuracy = 100. * correct / total
        val_accuracies.append(val_accuracy)

        # Save model checkpoint if this epoch has better validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(model, optimizer, epoch, save_path)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%")

    return train_losses, val_losses, val_accuracies


if __name__ == '__main__':
    set_random_seed(args.seed)
    args = get_args()

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
    data_loader, classes = build_dataset(args, is_train=True, trnsfrm=transform)
    model = vit.__dict__[args.model](img_size=224, num_classes=classes)
    model = model.to(device)
    optimizer = employ_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))
   
