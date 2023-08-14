import argparse
import sys
import torch
import torch.backends.cudnn as cudnn


from Architectures import vit 
from src.load_dataset import build_dataset
from src.transform import build_transform
from src.optimizer import employ_optimizer
from src.schedulers import init_lr_scheduler

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
    optimizer = employ_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))
   
