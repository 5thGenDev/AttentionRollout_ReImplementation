import argparse

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Model parameters
    parser.add_argument('--model', default='vit_base_16', type=str, 
                        choices=['vit_base_16', 'vit_large_32', 'vit_huge_14'], 
                        help="Name of architecture")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    
    ## Optimizers
    parser.add_argument("--optim", type=str, default="sgd", help="optimization algorithm (see optimizers.py)")
    parser.add_argument("--lr", default=0.0003, type=float, help="initial learning rate")
    parser.add_argument("--weight-decay", default=2e-05, type=float, help="weight decay")
    parser.add_argument("--epochs", default=4, type=int, help='4 is good starting point')
    parser.add_argument("--warmup_epochs", default=5, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Target LR at the end of optimization.")
    
    # sgd
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum factor for sgd and rmsprop")
    parser.add_argument("--sgd-dampening", default=0, type=float, help="sgd's dampening for momentum")
    parser.add_argument("--sgd-nesterov", action="store_true", help="whether to enable sgd's Nesterov momentum")
    
    # rmsprop
    parser.add_argument("--rmsprop-alpha", default=0.99, type=float, help="rmsprop's smoothing constant")
    
    # adam/amsgrad
    parser.add_argument("--adam-beta1", default=0.9, type=float, help="exponential decay rate for adam's first moment")
    parser.add_argument("--adam-beta2", default=0.999, type=float, help="exponential decay rate for adam's second moment")
    

    ## Dataset
    parser.add_argument('--data_set', default="Pets", type=str, choices=['CIFAR10', 'Flowers', 'Pets'], help='Name of the dataset.')
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument('--data_location', default='/path/to/dataset', type=str, help='Dataset location.')
    parser.add_argument('--output_dir', default="checkpoints/vit_small/trial", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument("--workers", default=4, type=int, help="number of data loading workers (tips: 4 or 8 times number of gpus)")

    
    ## GPUs
    parser.add_argument('--use_cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument("--gpu-devices", default="0", type=str, help="gpu device ids for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--use-avai-gpus", action="store_true", help="use available gpus instead of specified devices (useful when using managed clusters)")
        
    return args




def optimizer_kwargs(parsed_args):
    """
    Build kwargs for optimizer in optimizers.py from
    the parsed command-line arguments.
    """
    return {
        "optim": parsed_args.optim,
        "lr": parsed_args.lr,
        "weight_decay": parsed_args.weight_decay,
        "momentum": parsed_args.momentum,
        "sgd_dampening": parsed_args.sgd_dampening,
        "sgd_nesterov": parsed_args.sgd_nesterov,
        "rmsprop_alpha": parsed_args.rmsprop_alpha,
        "adam_beta1": parsed_args.adam_beta1,
        "adam_beta2": parsed_args.adam_beta2,
    }
