from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn

from utils.utils import AverageMeter
from utils.util import adjust_learning_rate, adjust_learning_rate_cosine, warmup_learning_rate, accuracy
from utils.util import set_optimizer, save_model
from models.resnet_linear import Fusion_R3D
from main import initailizing
from utils.temporal_transforms import TemporalSequentialCrop
from utils import spatial_transforms
from data.train_dataset import DAC
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

from opts import parse_args

def set_model(opt):
    model = Fusion_R3D(dash=R3D_MLP(opt.feature_dim, opt.model_depth, 
                        opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Dashboard.pth')),
                        rear=R3D_MLP(opt.feature_dim, opt.model_depth, 
                        opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Rear.pth')),
                        right=R3D_MLP(opt.feature_dim, opt.model_depth, 
                        opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Right.pth')),
                        with_classifier=True)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    return model, criterion
from data.cls_dataset import CLS_Train
def set_loader(opt):
    args = parse_args()
    crop_method, before_crop_duration = initailizing(args)
    temporal_transform = TemporalSequentialCrop(before_crop_duration, args.downsample)
    
    spatial_transform = spatial_transforms.Compose([
            spatial_transforms.Scale(args.sample_size),
            spatial_transforms.CenterCrop(args.sample_size),

            spatial_transforms.RandomRotate(),
            spatial_transforms.SaltImage(),
            spatial_transforms.Dropout(),
            spatial_transforms.ToTensor(args.norm_value),
            spatial_transforms.Normalize([0], [1])
        ])

    print("=================================Loading Driving Training Data!=================================")
    training_data = CLS_Train(root_path=args.root_path,
                                subset='train',
                                view=args.view,
                                sample_duration=before_crop_duration,
                                type='train',
                                spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform)

    
    print("=================================Loading Train Data!=================================")
    train_loader = DataLoader(
        training_data,
        batch_size=args.a_train_batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=False,
    )

    print("========================================Loading Validation Data========================================")
    val_spatial_transform = spatial_transforms.Compose([
        spatial_transforms.Scale(args.sample_size),
        spatial_transforms.CenterCrop(args.sample_size),
        spatial_transforms.ToTensor(args.norm_value),
        spatial_transforms.Normalize([0], [1])
    ])
    validation_data = CLS_Train(root_path=args.root_path,
                        subset='validation',
                        view=args.view,
                        sample_duration=args.sample_duration,
                        type=None,
                        spatial_transform=val_spatial_transform,
                        )
    
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=False,
    )
    num_val_data = validation_data.__len__()
    num_train_data = int(len(training_data))
    print(f'val_data len:{num_train_data}')
    print(f'val_data len:{num_val_data}')
    
    return train_loader, validation_loader


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg

from opts import parse_args
from models.resnet_linear import Fusion_R3D, R3D_MLP
from utils.util import Logger
def main():
    best_acc = 0
    opt = parse_args()
    
    train_loader, val_loader = set_loader(opt)
    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    model = torch.nn.DataParallel(model)
    logger = Logger(path=os.path.join(opt.log_folder, 'ce_train.log'), 
                    header=['epoch','epoch','train_loss','learning_rate','val_loss','val_acc' ], resume=opt.log_resume)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate_cosine(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        
        train_loss = loss.item()
        # evaluation
        loss, val_acc = validate(val_loader, model, criterion, opt)
        
        val_loss = val_loss.item()
        if val_acc > best_acc:
            best_acc = val_acc
        
        logger.log({'epoch':epoch,
                    'train_loss':train_loss,
                    'learning_rate':optimizer.param_groups[0]['lr'],
                    'val_loss':val_loss,
                    'val_acc':val_acc})
        
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
