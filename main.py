import os
import torch
import ast
from tqdm import tqdm
from opts import parse_args
from train import train
from test import test
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
from utils import spatial_transforms, temporal_transforms

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
    args.scales = [args.initial_scales]
    for i in range(1, args.n_scales):
        args.scales.append(args.scales[-1] * args.scale_step)
    assert args.train_crop in ['random', 'corner', 'center']
    if args.train_crop == 'random':
        crop_method = spatial_transforms.MultiScaleRandomCrop(args.scales, args.sample_size)
    elif args.train_crop == 'corner':
        crop_method = spatial_transforms.MultiScaleCornerCrop(args.scales, args.sample_size)
    elif args.train_crop == 'center':
        crop_method = spatial_transforms.MultiScaleCornerCrop(args.scales, args.sample_size, crop_positions=['c'])
    before_crop_duration = int(args.sample_duration * args.downsample)
    
    return crop_method, before_crop_duration
    
import random
import numpy as np

def main(index, args):
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    if index >= 0 and args.device.type == 'cuda':
        args.device = torch.device(f'cuda:{index}')

    # if args.distributed:
    #     args.dist_rank = args.dist_rank * args.ngpus_per_node + index
    #     dist.init_process_group(backend='nccl',
    #                             init_method=args.dist_url,
    #                             world_size=args.world_size,
    #                             rank=args.dist_rank)
    #     args.batch_size = int(args.batch_size / args.ngpus_per_node)
    #     args.n_threads = int(
    #         (args.n_threads + args.ngpus_per_node - 1) / args.ngpus_per_node)
    # args.is_master_node = not args.distributed or args.dist_rank == 0

    
    crop_method, crop_duration = initailizing(args)
    if args.nesterov:
        dampening = 0
    else:
        dampening = args.dampening
    
    if args.mode == 'train':
        train(args, before_crop_duration=crop_duration, crop_method=crop_method)
    elif args.mode == 'test':
        test(args)


import torch.multiprocessing as mp
import torch.distributed as dist

if __name__ == '__main__':
    print("Initializing..")
    from opts import parse_args
    args = parse_args()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    if args.use_cuda:
        cudnn.benchmark = True
    
    args.ngpus_per_node = torch.cuda.device_count()
    #print(int(os.environ["PMI_RANK"]))
    
    # if args.distributed:
    #     args.world_size = args.ngpus_per_node * args.world_size
    #     mp.spawn(main, nprocs=args.ngpus_per_node, args=(args,))
    main(-1, args)
