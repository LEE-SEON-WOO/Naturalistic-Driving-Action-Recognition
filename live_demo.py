import numpy as np
from utils import spatial_transforms
import os
from models.backbone import resnet
import torch.nn as nn
import torch
import cv2
from data.test_dataset import DAC_Test
from data.cls_dataset import CLS
from models.resnet_linear import Fusion_R3D
from models.prop_model import R3D_MLP
from opts import parse_args
from talco_test import cat_dataloaders, set_model

#==========================================================================================================
# This file is used after training and testing phase to demonstrate a running demo of the Driver Monitoring System
# with our contrastive approach.
#
# The demo is based on four well-trained 3D-ResNet-18 architectures, and four normal driving template vectors have been
# produced by these architectures at test time.
#
# When the code is running, a window will shows the current frame (8th frame of the test video clip) with predicted
# label; the path of the current frame, the similarity score, and predicted actions will be shown in terminal synchronized.
#
# If 'delay' is set to 0, the code runs one sample at a time. By pressing any key, the next sample will be processed
# If 'delay' is not 0, the code will process all samples from be beginning with a delay of 'delay' millisecond
#==========================================================================================================

#======================================Hyperparameters=====================================================
root_path = '../A2_slice/'  #root path of the dataset
show_which = 'Dashboard'  # show which view or modalities: {'Dashboard', Rear, Right}
threshold = 0.81  # the threshold
delay = 1  # The sample will be processed and shown with a delay of 'delay' ms.
           # If delay = 0, The code runs one sample at a time, press any key to process the next sample

sample_size = 112
sample_duration = 16
val_batch_size = 1
n_threads = 0
use_cuda = True
shortcut_type = 'A'
feature_dim = 512

print('========================================Loading Normal Vectors========================================')
normal_vec_front_d = np.load('./normvec/normal_vec_dashboard.npy')
normal_vec_front_ir = np.load('./normvec/normal_vec_rear.npy')
normal_vec_top_d = np.load('./normvec/normal_vec_right.npy')

normal_vec_front_d = torch.from_numpy(normal_vec_front_d)
normal_vec_front_ir = torch.from_numpy(normal_vec_front_ir)
normal_vec_top_d = torch.from_numpy(normal_vec_top_d)


if use_cuda:
    normal_vec_front_d = normal_vec_front_d.cuda()
    normal_vec_front_ir = normal_vec_front_ir.cuda()
    normal_vec_top_d = normal_vec_top_d.cuda()


val_spatial_transform = spatial_transforms.Compose([
    spatial_transforms.Scale(sample_size),
    spatial_transforms.CenterCrop(sample_size),
    spatial_transforms.ToTensor(255),
    spatial_transforms.Normalize([0], [1]),
])

print("===========================================Loading Test Data==========================================")
args = parse_args()

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

# test_data_front_d = DAD_Test(root_path=root_path,
#     subset='validation',
#     view='front_depth',
#     sample_duration=sample_duration,
#     type=None,
#     spatial_transform=val_spatial_transform,
# )
# test_loader_front_d = torch.utils.data.DataLoader(
#     test_data_front_d,
#     batch_size = val_batch_size,
#     shuffle = False,
#     num_workers = n_threads,
#     pin_memory = True,
# )
# num_val_data_front_d = test_data_front_d.__len__()
# print('Front depth view is done')

# test_data_front_ir = DAD_Test(root_path=root_path,
#     subset = 'validation',
#     view = 'front_IR',
#     sample_duration = sample_duration,
#     type = None,
#     spatial_transform = val_spatial_transform,
# )
# test_loader_front_ir = torch.utils.data.DataLoader(
#     test_data_front_ir,
#     batch_size = val_batch_size,
#     shuffle = False,
#     num_workers = n_threads,
#     pin_memory = True,
# )
# num_val_data_front_ir = test_data_front_ir.__len__()
# print('Front IR view is done')

# test_data_top_d = DAD_Test(root_path=root_path,
#     subset = 'validation',
#     view = 'top_depth',
#     sample_duration = sample_duration,
#     type = None,
#     spatial_transform = val_spatial_transform,
# )
# test_loader_top_d = torch.utils.data.DataLoader(
#     test_data_top_d,
#     batch_size = val_batch_size,
#     shuffle = False,
#     num_workers = n_threads,
#     pin_memory = True,
# )
# num_val_data_top_d = test_data_top_d.__len__()
# print('Top depth view is done')

# test_data_top_ir = DAD_Test(root_path=root_path,
#     subset = 'validation',
#     view = 'top_IR',
#     sample_duration = sample_duration,
#     type = None,
#     spatial_transform = val_spatial_transform,)

# test_loader_top_ir = torch.utils.data.DataLoader(
#     test_data_top_ir,
#     batch_size=val_batch_size,
#     shuffle=False,
#     num_workers=n_threads,
#     pin_memory=True,
# )
# num_val_data_top_ir = test_data_top_ir.__len__()
# print('Top IR view is done')
# assert num_val_data_front_d == num_val_data_front_ir == num_val_data_top_d

print('=============================================Loading Models===========================================')
#퓨전모델넣어

    
model = Fusion_R3D(dash=R3D_MLP(args.feature_dim, args.model_depth, 
                    opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Dashboard.pth')),
                    rear=R3D_MLP(args.feature_dim, args.model_depth, 
                    opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Rear.pth')),
                    right=R3D_MLP(args.feature_dim, args.model_depth, 
                    opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Right.pth')),
                    with_classifier=True)

model = torch.nn.DataParallel(model)

modelWeight, _ = set_model(args)

model.load_state_dict(state_dict = modelWeight)


# model_front_d = resnet.resnet18(
#     output_dim=feature_dim,
#     sample_size=sample_size,
#     sample_duration=sample_duration,
#     shortcut_type=shortcut_type,
# )

# model_front_ir = resnet.resnet18(
#     output_dim=feature_dim,
#     sample_size=sample_size,
#     sample_duration=sample_duration,
#     shortcut_type=shortcut_type,
# )

# model_top_d = resnet.resnet18(
#     output_dim=feature_dim,
#     sample_size=sample_size,
#     sample_duration=sample_duration,
#     shortcut_type=shortcut_type,
# )

# model_top_ir = resnet.resnet18(
#     output_dim=feature_dim,
#     sample_size=sample_size,
#     sample_duration=sample_duration,
#     shortcut_type=shortcut_type,
# )

# model_front_d = nn.DataParallel(model_front_d, device_ids=None)
# model_front_ir = nn.DataParallel(model_front_ir, device_ids=None)
# model_top_d = nn.DataParallel(model_top_d, device_ids=None)
# model_top_ir = nn.DataParallel(model_top_ir, device_ids=None)

# resume_path_front_d = './checkpoints/best_model_resnet_front_depth.pth'
# resume_path_front_ir = './checkpoints/best_model_resnet_front_IR.pth'
# resume_path_top_d = './checkpoints/best_model_resnet_top_depth.pth'
# resume_path_top_ir = './checkpoints/best_model_resnet_top_IR.pth'

# resume_checkpoint_front_d = torch.load(resume_path_front_d)
# resume_checkpoint_front_ir = torch.load(resume_path_front_ir)
# resume_checkpoint_top_d = torch.load(resume_path_top_d)
# resume_checkpoint_top_ir = torch.load(resume_path_top_ir)

# model_front_d.load_state_dict(resume_checkpoint_front_d['state_dict'])
# model_front_ir.load_state_dict(resume_checkpoint_front_ir['state_dict'])
# model_top_d.load_state_dict(resume_checkpoint_top_d['state_dict'])
# model_top_ir.load_state_dict(resume_checkpoint_top_ir['state_dict'])

# model_front_d.eval()
# model_front_ir.eval()
# model_top_d.eval()
# model_top_ir.eval()
model.eval()

print('===========================================Calculating Scores=========================================')#
#합친 데이터로더로 넣어
for batch, (data1, data2, data3) in enumerate(zip(test_loader)):
    if use_cuda:
        data1[0] = data1[0].cuda()
        data1[1] = data1[1].cuda()
        data2[0] = data2[0].cuda()
        data2[1] = data2[1].cuda()
        data3[0] = data3[0].cuda()
        data3[1] = data3[1].cuda()

    assert torch.sum(data1[1] == data2[1]) == torch.sum(data2[1] == data3[1]) == data1[1].size(0)

    out_1 = model(data1[0])[1].detach()
    out_2 = model(data2[0])[1].detach()
    out_3 = model(data3[0])[1].detach()

    sim_1 = torch.mm(out_1, normal_vec_front_d.t())
    sim_2 = torch.mm(out_2, normal_vec_front_ir.t())
    sim_3 = torch.mm(out_3, normal_vec_top_d.t())
    
    sim = round(torch.mean(torch.stack((sim_1, sim_2, sim_3), dim=0)).cpu().item(), 2)
    if sim >= threshold:
        action = 'Normal'
    else:
        action = 'Distracted'
        #classifier 넣기

    folder = int(batch // 60000) + 1
    subfolder = int((batch % 60000) // 10000) + 1

    index = (batch % 60000) % 10000
    img_path = os.path.join(root_path, 'val0'+str(folder)+'/rec'+str(subfolder)+'/'+show_which+'/img_'+str(index)+'.png')
    print(f'Img: {img_path} | score: {sim} | Action: {action}')

    img = cv2.imread(img_path)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score = 'Score: ' + str(sim)
    img = cv2.putText(img, score, (80, 20), font, 0.8, (0, 0, 255), 1)
    if action == 'Normal':
        img = cv2.putText(img, action, (93, 35), font, 0.8, (0, 0, 255), 1)
    elif action == 'Distracted':
        img = cv2.putText(img, action, (83, 35), font, 0.8, (0, 0, 255), 1)
    cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
    cv2.imshow('Demo', img)
    cv2.waitKey(delay)

cv2.destroyAllWindows()


