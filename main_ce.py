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
from utils.util import adjust_learning_rate_cosine, warmup_learning_rate, accuracy
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
from data.cls_dataset import CLS
class cat_dataloaders():
    """Class to concatenate multiple dataloaders"""

    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        len(self.dataloaders)

    def __iter__(self):
        self.loader_iter = []
        for data_loader in self.dataloaders:
            self.loader_iter.append(iter(data_loader))
        return self
    
    def __len__(self):
        return len(self.dataloaders[0])
    
    def __next__(self):
        out = []
        for data_iter in self.loader_iter:
            out.append(next(data_iter)) # may raise StopIteration
        return tuple(out)

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
    dash_dataset = CLS(root_path=args.root_path,
                                subset='train',
                                view='Dashboard',
                                sample_duration=before_crop_duration,
                                type='train',
                                spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform)
    dash_loader = DataLoader(
        dash_dataset,
        batch_size=args.a_train_batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=False,
    )
    rear_dataset = CLS(root_path=args.root_path,
                                subset='train',
                                view='Rear',
                                sample_duration=before_crop_duration,
                                type='train',
                                spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform)
    rear_loader = DataLoader(
        rear_dataset,
        batch_size=args.a_train_batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=False,
    )
    right_dataset = CLS(root_path=args.root_path,
                                subset='train',
                                view='Right',
                                sample_duration=before_crop_duration,
                                type='train',
                                spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform)
    right_loader = DataLoader(
        right_dataset,
        batch_size=args.a_train_batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=False,
    )
    train_loader = cat_dataloaders([dash_loader, rear_loader, right_loader])
    print("=================================Loading Train Data!=================================")
    
    
    print("========================================Loading Validation Data========================================")
    val_spatial_transform = spatial_transforms.Compose([
        spatial_transforms.Scale(args.sample_size),
        spatial_transforms.CenterCrop(args.sample_size),
        spatial_transforms.ToTensor(args.norm_value),
        spatial_transforms.Normalize([0], [1])
    ])
    val_dash = CLS(root_path=args.root_path,
                        subset='validation',
                        view='Dashboard',
                        sample_duration=args.sample_duration,
                        type=None,
                        spatial_transform=val_spatial_transform,
                        )
    val_rear = CLS(root_path=args.root_path,
                        subset='validation',
                        view='Rear',
                        sample_duration=args.sample_duration,
                        type=None,
                        spatial_transform=val_spatial_transform,
                        )
    val_right = CLS(root_path=args.root_path,
                        subset='validation',
                        view='Right',
                        sample_duration=args.sample_duration,
                        type=None,
                        spatial_transform=val_spatial_transform,
                        )
    
    val_dash_loader = DataLoader(
        val_dash,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=False,
    )
    val_rear_loader = DataLoader(
        val_rear,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=False,
    )
    val_right_loader = DataLoader(
        val_right,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=False,
    )
    validation_loader = cat_dataloaders([val_dash_loader, val_rear_loader, val_right_loader])
    num_val_data = val_dash_loader.__len__()
    num_train_data = int(len(dash_dataset))
    print(f'val_data len:{num_train_data}')
    print(f'val_data len:{num_val_data}')
    
    return train_loader, validation_loader


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.cuda()
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (dash, rear, right) in enumerate(train_loader):
        data_time.update(time.time() - end)
        dash_img, dash_label = dash
        rear_img, rear_label = rear
        right_img, right_label = right
        
        dash_img, dash_label = dash_img.cuda(non_blocking=True), dash_label.cuda(non_blocking=True)
        rear_img, right_img = rear_img.cuda(non_blocking=True), right_img.cuda(non_blocking=True)
        bsz = right_label.shape[0]
        

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        
        # compute loss
        output = model(dash_img, rear_img, right_img)
        loss = criterion(output, dash_label)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, dash_label, topk=(1, 5))
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
        for idx, (dash, rear, right) in enumerate(val_loader):

            dash_img, dash_label = dash
            rear_img, rear_label = rear
            right_img, right_label = right
            
            dash_img, dash_label = dash_img.cuda(non_blocking=True), dash_label.cuda(non_blocking=True)
            rear_img, right_img = rear_img.cuda(non_blocking=True), right_img.cuda(non_blocking=True)
            bsz = right_label.shape[0]
            
            
            # forward
            output = model(dash_img, rear_img, right_img)
            loss = criterion(output, dash_label)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, dash_label, topk=(1, 5))
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
        
        train_loss = loss
        # evaluation
        loss, val_acc = validate(val_loader, model, criterion, opt)
        
        val_loss = loss
            
        logger.log({'epoch':epoch,
                    'train_loss':train_loss,
                    'learning_rate':optimizer.param_groups[0]['lr'],
                    'val_loss':val_loss,
                    'val_acc':val_acc})
        
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
            
        if val_acc > best_acc:
            best_acc = val_acc
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
