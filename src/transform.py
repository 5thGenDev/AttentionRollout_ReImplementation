import PIL
from torchvision import datasets, transforms

# Input_size = 256
transform = transforms.Compose([
    transforms.Resize(size, interpolation=3),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(ImageNet_mean, ImageNet_std)])

# Input_size = 256
def build_transform(is_train, input_size):
    ImageNet_mean = (0.485, 0.456, 0.406)
    ImageNet_std = (0.229, 0.224, 0.225)

    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    
    # train transform
    if is_train:
        transform = transforms.Compose([
            # 3 = PIL.Image.BICUBIC
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=3), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(ImageNet_mean, ImageNet_mean)])
        return transform

    # eval transform
    transform = transforms.Compose([
        # for input_size = 256, size = 256
        transforms.Resize(size, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(ImageNet_mean, ImageNet_std)])
    return transform
