# Copyright (c) EEEM071, University of Surrey

import torch
import numpy 

def sequentialLR(
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
        optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr
    )

    # This is the schedule
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_epochs]
    )


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
