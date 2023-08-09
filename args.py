# Copyright (c) EEEM071, University of Surrey

import argparse

def args_parser():
    parser = argparse.ArgumentParser('SiT', add_help=False)

    # Reconstruction Parameters
    parser.add_argument('--drop_perc', type=float, default=0.5, help='Drop X percentage of the input image')
    parser.add_argument('--drop_replace', type=float, default=0.3, help='Drop X percentage of the input image')
    
    parser.add_argument('--drop_align', type=int, default=1, help='Align drop with patches; Set to patch size to align corruption with patches')
    parser.add_argument('--drop_type', type=str, default='zeros', help='Drop Type.')
    
    parser.add_argument('--lmbda', type=int, default=1, help='Scaling factor for the reconstruction loss')
    
    # Model parameters
    parser.add_argument('--model', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'], help="Name of architecture")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--weight_decay_end', type=float, default=0.1)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=800, type=int, help='Number of epochs of training.')

    parser.add_argument("--lr", default=0.5, type=float, help="Learning rate for SequentialLR. Current Pytorch SOTA")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="Number of epochs for SequentialLR warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="minLR for SequentialLR.")

  
    # Dataset
    parser.add_argument('--data_set', default='Pets', type=str, choices=['CIFAR10', 'Flowers', 'Pets'], help='Name of the dataset.')
    parser.add_argument('--data_location', default='/path/to/dataset', type=str, help='Dataset location.')

    parser.add_argument('--output_dir', default="checkpoints/vit_small/trial", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="set up distributed training")
    parser.add_argument("--local_rank", default=0, type=int)
    return parser
