'''Get dataset mean and std with PyTorch.'''
from __future__ import print_function

import logging
from datetime import datetime
from copy import deepcopy
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


import numpy as np
from data.train_dataset import DAC

if __name__ == '__main__':
        
        
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])
    from opts import parse_args
    args = parse_args()
    train_dataset = DAC(root_path=args.root_path,
                                subset='train',
                                view='Dashboard',
                                sample_duration=16,
                                type='normal',
                                spatial_transform=transform_train,
                                temporal_transform=None
                                )
    
    training_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=1, batch_size=1)
    print('%d training samples.' % len(training_loader))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h, w = 112, 112
    
    for batch_idx, (inputs) in enumerate(training_loader):
        
        inputs = inputs[0].to(device)
        if batch_idx == 0:
            bs, t, h, w = inputs.size(0), inputs.size(2), inputs.size(3), inputs.size(4)
            print(inputs.min(), inputs.max())
            chsum = inputs.sum(dim=(0, 2, 3, 4), keepdim=True) #bs, c, t, w, h
        else:
            chsum += inputs.sum(dim=(0, 2, 3, 4), keepdim=True)
    mean = chsum/len(train_dataset)/h/w/bs/t
    print('mean: %s' % mean.view(-1))
    
    chsum = None
    for batch_idx, (inputs) in enumerate(training_loader):
        inputs = inputs[0].to(device)
        if batch_idx == 0:
            chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3, 4), keepdim=True)
        else:
            chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3, 4), keepdim=True)
    std = torch.sqrt(chsum/(len(train_dataset)*bs*t * h * w - 1))
    print('std: %s' % std.view(-1))

    print('Done!')