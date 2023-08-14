import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model', default='vit_base_16', type=str, 
                        choices=['vit_base_16', 'vit_large_32', 'vit_huge_14'], 
                        help="Name of architecture")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.5, type=float, help="Learning rate.")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Target LR at the end of optimization.")

    # Dataset
    parser.add_argument('--data_set', default='Pets', type=str, choices=['CIFAR10', 'Flowers', 'Pets'], help='Name of the dataset.')
    parser.add_argument('--data_location', default='/path/to/dataset', type=str, help='Dataset location.')
    parser.add_argument('--output_dir', default="checkpoints/vit_small/trial", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--num_workers', default=8, type=int, help='4-8x # of GPUs.')

    # GPUs
    parser.add_argument('--use_cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument("--gpu-devices", default="0", type=str, help="gpu device ids for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--use-avai-gpus", action="store_true", help="use available gpus instead of specified devices (useful when using managed clusters)")
    
    args = parser.parse_args()
    
    if not args.use_avai_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")
        
    return args
