import torch
import csv
import math
import pandas as pd
import random
from opts import parse_args
import numpy as np
from collections import OrderedDict
from opts import parse_args
from utils import spatial_transforms
from tqdm import tqdm
from models.resnet_linear import Fusion_R3D
from models.prop_model import R3D_MLP
from data.cls_dataset import CLS
from postprocessing import postprocessing

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

def set_model(opt):
    model = torch.load('./checkpoints/ckpt_epoch_best_2.14747_4.pth')
    state_dict = model['model']
    criterion =''
    return state_dict, criterion

def set_loader(args):             
    print("========================================Loading Test Data========================================") 
    test_spatial_transform = spatial_transforms.Compose([
    spatial_transforms.Scale(args.sample_size),
    spatial_transforms.CenterCrop(args.sample_size),
    spatial_transforms.ToTensor(args.norm_value),
    spatial_transforms.Normalize([0], [1])
    ])
    
    test_data_dashboard = CLS(root_path=args.root_path,
                                subset='test',
                                view='Dashboard',
                                type='test',
                                spatial_transform=test_spatial_transform
                                )
    
    test_loader_dashboard = torch.utils.data.DataLoader(
        test_data_dashboard,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=False)
    num_val_data_dashboard = test_data_dashboard.__len__()
    print('Dashboard view is done')

    test_data_rear = CLS(root_path=args.root_path,
                                subset='test',
                                view='Rear',
                                type='test',
                                spatial_transform=test_spatial_transform
                                )
    
    test_loader_rear = torch.utils.data.DataLoader(
                        test_data_rear,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=args.n_threads,
                        pin_memory=False)
    num_val_data_rear = test_data_rear.__len__()
    print('Rear view is done')

    test_data_right = CLS(root_path=args.root_path,
                                subset='test',
                                view='Right',
                                type='test',
                                spatial_transform=test_spatial_transform
                                )
    
    test_loader_right = torch.utils.data.DataLoader(
        test_data_right,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=False,
    )
    num_val_data_right = test_data_right.__len__()
    print('Right view is done')
    assert num_val_data_dashboard == num_val_data_rear == num_val_data_right
    test_loader = cat_dataloaders([test_loader_dashboard, test_loader_rear, test_loader_right])

    return test_loader

def max_k(output, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        

        _, pred = output.topk(maxk, 1, True, True)
        return pred.cpu().numpy().reshape(1, -1).tolist()
import os
from collections import defaultdict
def test(test_loader, model):
    model.cuda()
    model.eval()
    outputs = defaultdict(list)
    with torch.no_grad():
        for _, (dash, rear, right) in tqdm(enumerate(test_loader)):
            dash_img, inform = dash
            rear_img, _ = rear
            right_img, _ = right
            dash_img = dash_img.cuda(non_blocking=True)
            rear_img, right_img = rear_img.cuda(non_blocking=True), right_img.cuda(non_blocking=True)
            # forward
            output = model(dash_img, rear_img, right_img)
            # outputs.append([inform[0][13:],output,inform[1]])
            max_top1 = max_k(output, topk=(1,))
            video_id = [int(path.split(os.sep)[-1]) for path in inform[0]]
            time = inform[1].numpy().tolist()
            outputs['video_id'].extend(video_id)
            outputs['activity_id'].extend(max_top1[0])
            outputs['time'].extend(time)

    os.makedirs('./output', exist_ok=True)
    csv_path = './output/output.csv'
    writer = pd.DataFrame(outputs)
    writer.to_csv(csv_path, sep=' ', index=False, header=None, line_terminator = '\n')

from test1 import save_aiformat
def main(index, args):
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    
    if index >= 0 and args.device.type == 'cuda':
        args.device = torch.device(f'cuda:{index}')
    opt = parse_args()
    
    model = Fusion_R3D(dash=R3D_MLP(opt.feature_dim, opt.model_depth, 
                        opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Dashboard.pth')),
                        rear=R3D_MLP(opt.feature_dim, opt.model_depth, 
                        opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Rear.pth')),
                        right=R3D_MLP(opt.feature_dim, opt.model_depth, 
                        opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Right.pth')),
                        with_classifier=True)
    model = torch.nn.DataParallel(model)
    
    modelWeight, _ = set_model(opt)
    
    model.load_state_dict(state_dict = modelWeight)
    
    test_loader = set_loader(args)

    test(test_loader, model)
    
    
    #postprocessing('./output/output.csv','./output/processedOutput.csv')
    save_aiformat(load_path='./output/output.csv', save_path='./output/last.txt')
    
    
if __name__ == "__main__":
    args = parse_args()
    main(-1,args)