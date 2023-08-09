import warnings
warnings.filterwarnings("ignore")

import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

#from datasets import load_dataset, datasets_utils
from src.dataset_loader import build_dataset
from args import get_args_parser

import utils
import vision_transformer as vits
from vision_transformer import CLSHead, RECHead
import torchvision


# Global vars
parser = get_args_parser()
args = parser.parse_args()


# replace from other images
class collate_batch(object): 
    def __init__(self, drop_replace=0., drop_align=1):
        self.drop_replace = drop_replace
        self.drop_align = drop_align
        
    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        
        if self.drop_replace > 0:
            batch[0][1][0], batch[0][2][0] = datasets_utils.GMML_replace_list(batch[0][0][0], batch[0][1][0], batch[0][2][0],
                                                                            max_replace=self.drop_replace, align=self.drop_align)
            batch[0][1][1], batch[0][2][1] = datasets_utils.GMML_replace_list(batch[0][0][1], batch[0][1][1], batch[0][2][1],
                                                                            max_replace=self.drop_replace, align=self.drop_align)
        
        return batch
    
    
def train_SiT(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    cudnn.benchmark = True

    # prepare dataset
    # Finetuned dataset has to be preprocessed exactly as pretrained dataset
    transform = datasets_utils.DataAugmentationSiT(args)

    # Need to fix this 
    dataset, _ = build_dataset(args, True, trnsfrm=transform, training_mode = 'SSL')
    
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(dataset,
        sampler=sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, 
        collate_fn=collate_batch(args.drop_replace, args.drop_align))
    print(f"Data loaded: there are {len(dataset)} images.")

    # building networks 
    student = vits.__dict__[args.model](drop_path_rate=args.drop_path_rate)
    teacher = vits.__dict__[args.model]()
    embed_dim = student.embed_dim

    student = FullPipline(student, CLSHead(embed_dim, args.out_dim), RECHead(embed_dim))
    teacher = FullPipline(teacher, CLSHead(embed_dim, args.out_dim), RECHead(embed_dim))
    student, teacher = student.cuda(), teacher.cuda()
    
    # synchronize batch norms
    student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    # we need DDP wrapper to have synchro batch norms working...
    teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
    teacher_without_ddp = teacher.module

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.model} network.")

    # preparing SimCLR loss
    simclr_loss = SimCLR(args.simclr_temp).cuda()

    # preparing optimizer 
    optimizer = torch.optim.AdamW(utils.get_params_groups(student))  # to use with ViTs

    # for mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # init schedulers 
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256., 
        args.min_lr, args.epochs, len(data_loader), warmup_epochs=args.warmup_epochs)
    
    wd_schedule = utils.cosine_scheduler( args.weight_decay,
        args.weight_decay_end, args.epochs, len(data_loader))
    
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))

    # Resume training 
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student, teacher=teacher,
        optimizer=optimizer, fp16_scaler=fp16_scaler)
    
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Training ..")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # Training
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, simclr_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args)

        # logs
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1, 'args': args}
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, simclr_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    
    save_recon = os.path.join(args.output_dir, 'reconstruction_samples')
    Path(save_recon).mkdir(parents=True, exist_ok=True)
    bz = args.batch_size
    plot_ = True
    
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    for it, ((clean_crops, corrupted_crops, masks_crops), _) in enumerate(metric_logger.log_every(data_loader, 100, header)):

        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        clean_crops = [im.cuda(non_blocking=True) for im in clean_crops]
        corrupted_crops = [im.cuda(non_blocking=True) for im in corrupted_crops]
        masks_crops = [im.cuda(non_blocking=True) for im in masks_crops]
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            t_cls, _ = teacher(torch.cat(clean_crops[0:]), recons=False) 
            s_cls, s_recons = student(torch.cat(corrupted_crops[0:]))
            
            c_loss = simclr_loss(s_cls, t_cls, epoch)
            
            #-------------------------------------------------
            recloss = F.l1_loss(s_recons, torch.cat(clean_crops[0:]), reduction='none')
            r_loss = recloss[torch.cat(masks_crops[0:2])==1].mean() 
                
            if plot_==True and utils.is_main_process():
                plot_ = False
                #validating: check the reconstructed images
                print_out = save_recon + '/epoch_' + str(epoch).zfill(5)  + '.jpg' 
                imagesToPrint = torch.cat([clean_crops[0][0: min(15, bz)].cpu(),  corrupted_crops[0][0: min(15, bz)].cpu(),
                                       s_recons[0: min(15, bz)].cpu(), masks_crops[0][0: min(15, bz)].cpu()], dim=0)
                torchvision.utils.save_image(imagesToPrint, print_out, nrow=min(15, bz), normalize=True, range=(-1, 1))
            
            
            loss = c_loss + args.lmbda * r_loss
            
           

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)

            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  
                param_norms = utils.clip_gradients(student, args.clip_grad)

            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(c_loss=c_loss.item())
        metric_logger.update(r_loss=r_loss.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class SimCLR(nn.Module):
    def __init__(self, temp=0.2):
        super().__init__()
        
        self.temp = temp
        
    def contrastive_loss(self, q, k):
        
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        # gather all targets
        k = concat_all_gather(k)
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.temp
        N = logits.shape[0] 
        
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.temp)

    def forward(self, student_output, teacher_output, epoch):

        student_out = student_output
        student_out = student_out.chunk(2)

        teacher_out = teacher_output 
        teacher_out = teacher_out.detach().chunk(2)

        return self.contrastive_loss(student_out[0], teacher_out[1]) + self.contrastive_loss(student_out[1], teacher_out[0])


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class FullPipline(nn.Module):
    def __init__(self, backbone, head, head_recons):
        super(FullPipline, self).__init__()

        
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.head_recons = head_recons

    def forward(self, x, recons=True):
        _out = self.backbone(x)
        
        if recons==True:
            return self.head(_out[:, 0]), self.head_recons(_out[:, 1:])
        else:
            return self.head(_out[:, 0]), None


if __name__ == '__main__':
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_SiT(args)
