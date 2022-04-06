import os
import torch
import csv
import math
import random
from opts import parse_args
import numpy as np
from opts import parse_args
from models.resnet_linear import Fusion_R3D
import torch.multiprocessing as mp
import torch.distributed as dist
from models.prop_model import R3D_MLP
from models.resnet_linear import Fusion_R3D
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

def test(test_loader, model):
    model.eval()
    
    with open('./output.csv','w') as f:
        fieldnames = ['video_id', 'activity_id', 'start_time', 'end_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        with torch.no_grad():
            outputs = []
            for idx, (dash, rear, right) in enumerate(test_loader):

                    dash_img, label = dash
                    rear_img, _ = rear
                    _, right_label = right
                    
                    dash_img, dash_label = dash_img.cuda(non_blocking=True), label.cuda(non_blocking=True)
                    rear_img, right_img = rear_img.cuda(non_blocking=True), right_img.cuda(non_blocking=True)
                    bsz = label.shape[0]
                    
                    
                    # forward
                    output = model(dash_img, rear_img, right_img)
                    outputs.append(output)
                    
            writer.writerow(outputs)
            f.close()
    


def set_loader(args):             # 데이터 로더 3개 어떻게 묶을지
    print("========================================Loading Test Data========================================") 
    test_data_dashboard = CLS(root_path=args.root_path,
                                view='Dashboard',)
    
    test_loader_dashboard = torch.utils.data.DataLoader(
        test_data_dashboard,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=False)
    num_val_data_dashboard = test_data_dashboard.__len__()
    print('Dashboard view is done')

    test_data_rear = CLS(root_path=args.root_path,
                                view='Rear',)
    
    test_loader_rear = torch.utils.data.DataLoader(
                        test_data_rear,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=args.n_threads,
                        pin_memory=False)
    num_val_data_rear = test_data_rear.__len__()
    print('Rear view is done')

    test_data_right = CLS(root_path=args.root_path,
                                view='Right',)
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

def main(index, args):
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    
    if index >= 0 and args.device.type == 'cuda':
        args.device = torch.device(f'cuda:{index}')

    model = set_model()
    test_loader = set_loader(args)

    test(test_loader, model)
    
    
    
if __name__ == "__main__":
    args = parse_args()
    main(-1,args)