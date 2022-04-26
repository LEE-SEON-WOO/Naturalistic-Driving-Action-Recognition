import os
import torch
from opts import parse_args
from train import train
from test import test
from torch.backends import cudnn

def initailizing(args):
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    if not os.path.exists(args.normvec_folder):
        os.makedirs(args.normvec_folder)
    if not os.path.exists(args.score_folder):
        os.makedirs(args.score_folder)
    torch.manual_seed(args.manual_seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.manual_seed)

import random
import numpy as np

def main(index, args):
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    if index >= 0 and args.device.type == 'cuda':
        args.device = torch.device(f'cuda:{index}')

    initailizing(args)
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


import torch.multiprocessing as mp
import torch.distributed as dist

if __name__ == '__main__':
    print("Initializing..")
    from opts import parse_args
    args = parse_args(root_path='../A1/newFrame/')
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    if args.use_cuda:
        cudnn.benchmark = True
    args.ngpus_per_node = torch.cuda.device_count()
    main(-1, args)
