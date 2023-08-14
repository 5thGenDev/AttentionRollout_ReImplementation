# Copyright (c) EEEM071, University of Surrey

import torch

def init_lr_scheduler(
    optimizer,
    lr_scheduler,  # learning rate scheduler
    epochs,
    stepsize=[20, 40],  # step size to decay learning rate
    gamma=0.1,  # learning rate decay
):
    '''
    Read this line of code: https://github.com/pytorch/vision/blob/main/references/classification/train.py#L304
    and: https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#lr-optimizations
    '''
    lr_warmup_epochs = 5
    lr_warmup_decay = 0.01
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=lr_warmup_decay, total_iters=lr_warmup_epochs
    )

    '''
    Read this line of code: https://github.com/pytorch/vision/blob/main/references/classification/train.py#L291
    and: https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#:~:text=decaying%20the%20LR%20up%20to%20zero
    '''
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - lr_warmup_epochs, eta_min=0
    )
    
    if lr_scheduler == "single_step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize[0], gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "sequential":
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[lr_warmup_epochs]
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")
