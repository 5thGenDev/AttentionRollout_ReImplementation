import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('SiT', add_help=False) 

    # Model parameters
    parser.add_argument('--model', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'], help="Name of architecture")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    

    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=0.04)
    
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs of training.')

    parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate.")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Target LR at the end of optimization.")
    

    # Dataset
    parser.add_argument('--data_set', default='Pets', type=str, choices=['CIFAR10', 'Flowers', 'Pets'], help='Name of the dataset.')
    parser.add_argument('--data_location', default='/path/to/dataset', type=str, help='Dataset location.')

    parser.add_argument('--output_dir', default="checkpoints/vit_small/trial", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--num_workers', default=8, type=int, help='4-8x # of GPUs.')
    parser.add_argument("--dist_url", default="env://", type=str, help="set up distributed training")
    return parser
