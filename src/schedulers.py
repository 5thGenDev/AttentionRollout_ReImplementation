# Copyright (c) EEEM071, University of Surrey

import torch


def sequential(
    optimizer,
    epochs,
    warmup_epochs=5,
    min_lr=0,
):
    '''
    Read this line of code: https://github.com/pytorch/vision/blob/main/references/classification/train.py#L304
    and: https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#lr-optimizations
    for warmup_epoch and warmup_decay
    '''
    lr_warmup_decay = 0.01
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=lr_warmup_decay, total_iters=warmup_epochs
    )

    '''
    Read this line of code: https://github.com/pytorch/vision/blob/main/references/classification/train.py#L291
    and: https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#:~:text=decaying%20the%20LR%20up%20to%20zero
    '''
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - lr_warmup_epochs, eta_min=min_lr
    )
    
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_epochs]
    )

