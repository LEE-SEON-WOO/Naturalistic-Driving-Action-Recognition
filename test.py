import os
import torch
from utils import spatial_transforms
import numpy as np
from models.prop_model import R3D_MLP
from data.test_dataset import DAC_Test
from data.train_dataset import DAC
from utils.util import get_normal_vector, cal_score, get_score, get_fusion_label, evaluate, post_process

def test(args):
    if not os.path.exists(args.normvec_folder):
        os.makedirs(args.normvec_folder)
    score_folder = './score/'
    if not os.path.exists(score_folder):
        os.makedirs(score_folder)
    args.pre_train_model = False

    model_dashboard = R3D_MLP(feature_dim=args.feature_dim, model_depth=args.model_depth, opt=args)
    model_rear = R3D_MLP(feature_dim=args.feature_dim, model_depth=args.model_depth, opt=args)
    model_right = R3D_MLP(feature_dim=args.feature_dim, model_depth=args.model_depth, opt=args)

    resume_path_dashboard = './checkpoints/best_model_' + args.model_type + '_Dashboard.pth'
    resume_path_rear = './checkpoints/best_model_' + args.model_type + '_Rear.pth'
    resume_path_right = './checkpoints/best_model_' + args.model_type + '_Right.pth'

    resume_checkpoint_dashboard = torch.load(resume_path_dashboard)
    resume_checkpoint_rear = torch.load(resume_path_rear)
    resume_checkpoint_right = torch.load(resume_path_right)
    
    model_dashboard.load_state_dict(resume_checkpoint_dashboard['state_dict'], strict=False)
    model_rear.load_state_dict(resume_checkpoint_rear['state_dict'], strict=False)
    model_right.load_state_dict(resume_checkpoint_right['state_dict'], strict=False)
        
    model_dashboard.eval()
    model_rear.eval()
    model_right.eval()

    val_spatial_transform = spatial_transforms.Compose([
        spatial_transforms.Scale(args.sample_size),
        spatial_transforms.CenterCrop(args.sample_size),
        spatial_transforms.ToTensor(args.norm_value),
        spatial_transforms.Normalize([0], [1])])

    print("========================================Loading Test Data========================================")  # Test 데이타 수정 필요한 부분
    test_data_dashboard = DAC_Test(root_path=args.root_path,
                                subset='validation',
                                view='Dashboard',
                                random_state=args.manual_seed,
                                sample_duration=args.sample_duration,
                                type=None,
                                spatial_transform=val_spatial_transform)
    test_loader_dashboard = torch.utils.data.DataLoader(
        test_data_dashboard,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=False)
    num_val_data_dashboard = test_data_dashboard.__len__()
    print('Dashboard view is done')

    test_data_rear = DAC_Test(root_path=args.root_path,
                            subset='validation',
                            view='Rear',
                            sample_duration=args.sample_duration,
                            random_state=args.manual_seed,
                            type=None,
                            spatial_transform=val_spatial_transform)
    test_loader_rear = torch.utils.data.DataLoader(
                        test_data_rear,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=args.n_threads,
                        pin_memory=False)
    num_val_data_rear = test_data_rear.__len__()
    print('Rear view is done')

    test_data_right = DAC_Test(root_path=args.root_path,
                                subset='validation',
                                view='Right',
                                sample_duration=args.sample_duration,
                                random_state=args.manual_seed,
                                type=None,
                                spatial_transform=val_spatial_transform,
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

    print("==========================================Loading Normal Data==========================================")
    training_normal_dashboard = DAC(root_path=args.root_path,
                                    subset='train',
                                    view='Dashboard',
                                    sample_duration=args.sample_duration,
                                    random_state=args.manual_seed,
                                    type='normal',
                                    spatial_transform=val_spatial_transform)

    training_normal_size = int(len(training_normal_dashboard) * args.n_split_ratio)
    training_normal_dashboard = torch.utils.data.Subset(training_normal_dashboard,
                                                        np.arange(training_normal_size))

    train_normal_loader_for_test_dashboard = torch.utils.data.DataLoader(
        training_normal_dashboard,
        batch_size=args.cal_vec_batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=False,
    )
    print(f'Dash board view is done (size: {len(training_normal_dashboard)})')

    training_normal_data_rear = DAC(root_path=args.root_path,
                                        subset='train',
                                        view='Rear',
                                        sample_duration=args.sample_duration,
                                        random_state=args.manual_seed,
                                        type='normal',
                                        spatial_transform=val_spatial_transform)

    training_normal_size = int(len(training_normal_data_rear) * args.n_split_ratio)
    training_normal_data_rear = torch.utils.data.Subset(training_normal_data_rear,
                                                            np.arange(training_normal_size))

    train_normal_loader_for_test_rear = torch.utils.data.DataLoader(
        training_normal_data_rear,
        batch_size=args.cal_vec_batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=False)
    print(f'Rear view is done (size: {len(training_normal_data_rear)})')

    training_normal_data_right = DAC(root_path=args.root_path,
                                    subset='train',
                                    view='Right',
                                    sample_duration=args.sample_duration,
                                    random_state=args.manual_seed,
                                    type='normal',
                                    spatial_transform=val_spatial_transform)

    training_normal_size = int(len(training_normal_data_right) * args.n_split_ratio)
    training_normal_data_right = torch.utils.data.Subset(training_normal_data_right,
                                                        np.arange(training_normal_size))

    train_normal_loader_for_test_right = torch.utils.data.DataLoader(
        training_normal_data_right,
        batch_size=args.cal_vec_batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=False,
    )
    print(f'Right view is done (size: {len(training_normal_data_right)})')

    print("============================================START EVALUATING============================================")
    normal_vec_dashboard = get_normal_vector(model_dashboard, train_normal_loader_for_test_dashboard,
                                            args.cal_vec_batch_size,
                                            args.feature_dim,
                                            args.use_cuda)
    np.save(os.path.join(args.normvec_folder, 'normal_vec_dashboard.npy'), normal_vec_dashboard.cpu().numpy())

    normal_vec_rear = get_normal_vector(model_rear, train_normal_loader_for_test_rear,
                                            args.cal_vec_batch_size,
                                            args.feature_dim,
                                            args.use_cuda)
    np.save(os.path.join(args.normvec_folder, 'normal_vec_rear.npy'), normal_vec_rear.cpu().numpy())

    normal_vec_right = get_normal_vector(model_right, train_normal_loader_for_test_right, args.cal_vec_batch_size,
                                            args.feature_dim,
                                            args.use_cuda)
    np.save(os.path.join(args.normvec_folder, 'normal_vec_right.npy'), normal_vec_right.cpu().numpy())

    cal_score(model_dashboard, model_rear, model_right, 
                normal_vec_dashboard, normal_vec_rear, normal_vec_right, 
                test_loader_dashboard, test_loader_rear, test_loader_right, 
                score_folder, args.use_cuda)

    gt = get_fusion_label(os.path.join(args.root_path, 'LABEL.csv'))

    hashmap = {'Dashboard': 'dashboard',
                'Rear': 'rear',
                'Right': 'Right',
                
                'fusion_Dashboard_Rear': 'fu_dash_rear',
                'fusion_Dashboard_Right': 'fu_dash_right',
                'fusion_Rear_Right': 'fu_rear_right',
                
                'fusion_all': 'fu_all'
                }

    for mode, mode_name in hashmap.items():
        score = get_score(score_folder, mode)
        best_acc, best_threshold, AUC = evaluate(score, gt, False)
        print(
            f'Mode: {mode_name}:      Best Acc: {round(best_acc, 2)} | Threshold: {round(best_threshold, 2)} | AUC: {round(AUC, 4)}')
        score = post_process(score, args.window_size)
        best_acc, best_threshold, AUC = evaluate(score, gt, False)
        print(
            f'View: {mode_name}(post-processed):       Best Acc: {round(best_acc, 2)} | Threshold: {round(best_threshold, 2)} | AUC: {round(AUC, 4)} \n')


