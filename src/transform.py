from torchvision import datasets, transforms

# input_size = desired size
def build_transform(is_train, input_size=224):
    ImageNet_mean = (0.485, 0.456, 0.406)
    ImageNet_std = (0.229, 0.224, 0.225)

    # train transform
    if is_train:
        transform = transforms.Compose([
            # 3 = PIL.Image.BICUBIC
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(ImageNet_mean, ImageNet_mean),
        ])
        return transform

    # eval transform
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(ImageNet_mean, ImageNet_std),
    ])
    return transform
