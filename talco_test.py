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

def set_model(ckpt_path):
    model = torch.load(ckpt_path)
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
        

        prob, pred = output.topk(maxk, 1, True, True)
        
    return prob.cpu().numpy().reshape(-1, 1), pred.cpu().numpy().reshape(-1, 1)
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
            
            # input shape is (BS, 1, 16, 112, 112) 거진 0.5초당 한번 계산
            output = model(dash_img, rear_img, right_img)
            # output shape is  (BS 18)
            
            probs, max_top1 = max_k(output, topk=(1,)) 
            
            # 확률값들이랑 가장 큰값 하나를 구함
            threshold = np.mean(output.cpu().numpy(), axis=1)
            # 확률값 평균을 내서 임계치를 구함
            
            action_tag = np.zeros(max_top1.shape) #batch size만큼 zero matrix 할당
            probs = probs.reshape(-1)
            threshold = threshold.reshape(-1)
            max_top1 = max_top1.reshape(-1)
            action_tag[probs >=threshold] = 1 # 임계치보다 큰 값들을 1으로 채움
            #(bs, 1)
            activities_idx = []
            startings = []
            endings = []
            
            for i in range(len(action_tag)): #bs만큼 돌면서
                if action_tag[i] ==1: #만약 1이면
                    activities_idx.append(max_top1[i])
                    start = i
                    end = i+1
                    startings.append(start)
                    endings.append(end)
            
            outputs['start'].extend(startings)
            outputs['end'].extend(endings)
            outputs['activity_id'].extend(activities_idx)
            


    os.makedirs('./output', exist_ok=True)
    csv_path = './output/output.csv'
    writer = pd.DataFrame(outputs) # 'start' 'end'  'activity_id'
    writer.to_csv(csv_path, sep=' ', index=False, header=None, line_terminator = '\n')
    
import math
import numpy as np
import pandas as pd

def save_aiformat(load_path, save_path):
    output= np.loadtxt(load_path, delimiter=' ')
    # indx = np.array([i for i in range(output.shape[0])])
    mask = np.equal(np.roll(output[:,:], -1, axis=0), output[:,:]) #달라졌는지 체크
    #output = video, class ,time
    
    r_mask = np.logical_and(mask[:, 0], mask[:, 1])
    # out = output[r_mask].astype(int)

    r_n_mask = np.logical_not(r_mask)

    res = np.hstack([(mask[r_n_mask]), output[r_n_mask]]).astype(int)
    # res = pd.DataFrame(columns=['Equal1','Equal2', index, 'video_ID', 'activityid', 'time'])
    start_time = 0
    outputs= []
    #입력비디오, Class간비교한거(Bool), 인덱스, 비디오아이디, 예측값, 프레임끝번호
    #print(res[:10])
    #exit()
    for item in res:
        outputs.append([item[3],item[4], int(start_time/30), int((item[5]+15)/30)])
        if not item[0]: #입력 비디오가 달라졌는지?
            start_time = 0 #입력 시간이 0부터 시작
        elif not item[1]: #이상행동 클래스가 달라졌는지?
            start_time = item[5]+16 #입력 시간이 0부터 시작
        else:
            pass
    
    
    np.savetxt(save_path, np.array(outputs).astype(int), delimiter=' ', fmt='%d')
    
    df_total = pd.read_csv(save_path, delimiter=' ', header=None)
    df_total = df_total[df_total[2]!=0] # 0 클래스인것 제거
    df_total = df_total[df_total[3] - df_total[2] >6]
    df_total = df_total.sort_values(by=[0, 1])
    df_total.to_csv('./output/last.txt', sep=' ', index = False, header=None, line_terminator = '\n')
    
def main(index, args, ckpt_path):
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    
    if index >= 0 and args.device.type == 'cuda':
        args.device = torch.device(f'cuda:{index}')
    opt = args
    
    model = Fusion_R3D(dash=R3D_MLP(opt.feature_dim, opt.model_depth, 
                        opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Dashboard.pth')),
                        rear=R3D_MLP(opt.feature_dim, opt.model_depth, 
                        opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Rear.pth')),
                        right=R3D_MLP(opt.feature_dim, opt.model_depth, 
                        opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Right.pth')),
                        with_classifier=True)
    model = torch.nn.DataParallel(model)
    
    modelWeight, _ = set_model(ckpt_path)
    
    model.load_state_dict(state_dict = modelWeight)
    
    test_loader = set_loader(args)
    
    test(test_loader, model)
    
    
    # postprocessing('./output/output.csv','./output/processedOutput.csv')
    save_aiformat(load_path='./output/output.csv', save_path='./output/last.txt')
    
if __name__ == "__main__":
    args = parse_args(root_path='../A2_slice/',
                    mode='test')
    main(-1,args, ckpt_path='./checkpoints/ckpt_epoch_best_2.37050_1.pth')