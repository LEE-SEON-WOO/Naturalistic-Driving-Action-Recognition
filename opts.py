import argparse
from pathlib import Path



import argparse
import ast

def parse_args(pretrain_path:str='../pretrained/r3d18_K_200ep.pth', 
                n_classes:int=18,
                n_pretrain_classes:int=1139,
                n_input_channels:int=3,
                model:str='resnet',
                model_depth:int=50,
                conv1_t_size:int=7,
                conv1_t_stride:int=1,
                resnet_shortcut:str='B',
                resnet_widen_factor:float=1.0,
                wide_resnet_k:int=2,
                resnext_cardinality:int=32,
                manual_seed:int=1,
                feature_dim=128,
                sample_size=112,
                output_topk:int=5,
                sample_duration=16,
                mode:str='test',
                root_path:str='../A2_slice/',
                input_type:str='rgb'):
    parser = argparse.ArgumentParser(description='DAD training on Videos')
    #Train/Test
    parser.add_argument('--root_path', default=root_path, type=str, help='root path of the dataset')
    # parser.add_argument('--root_path', default='../A2_slice/', type=str, help='root path of the dataset')
    parser.add_argument('--mode', default=mode, type=str, help='train | test(validation)')
    parser.add_argument('--feature_dim', default=feature_dim, type=int, help='To which dimension will video clip be embedded')
    parser.add_argument('--sample_duration', default=sample_duration, type=int, help='Temporal duration of each video clip')
    parser.add_argument('--sample_size', default=sample_size, type=int, help='Height and width of inputs')
    parser.add_argument('--model_type', default='resnet', type=str, help='so far only resnet')
    parser.add_argument('--shortcut_type', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--pre_train_model', default=True, type=ast.literal_eval, help='Whether use pre-trained model')
    parser.add_argument('--n_train_batch_size', default=32, type=int, help='Batch Size for normal training data')
    parser.add_argument('--a_train_batch_size', default=16, type=int, help='Batch Size for anormal training data')
    parser.add_argument('--val_batch_size', default=32, type=int, help='Batch Size for validation data')
    parser.add_argument('--checkpoint_folder', default='./checkpoints/', type=str, help='folder to store checkpoints')
    parser.add_argument('--log_folder', default='./logs/', type=str, help='folder to store log files')
    parser.add_argument('--log_resume', default=False, type=ast.literal_eval, help='True|False: a flag controlling whether to create a new log file')
    parser.add_argument('--normvec_folder', default='./normvec/', type=str, help='folder to store norm vectors')
    parser.add_argument('--score_folder', default='./score/', type=str, help='folder to store scores')
    parser.add_argument('--val_step', default=1, type=int, help='validate per val_step epochs')
    parser.add_argument('--save_step', default=10, type=int, help='checkpoint will be saved every save_step epochs')
    parser.add_argument('--resume_path', default='', type=str, help='path of previously trained model')
    parser.add_argument('--cosine', default=True, type=bool, help='using cosine annealing')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--warm', default=False, type=bool, help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', default=10, type=int, help='warm epochs for training')
    parser.add_argument('--warmup_from', default=0.01, type=float, help='warm epochs for training')
    parser.add_argument('--print_freq', default=1, type=int, help='print frequency when u train/test gui interface')
    parser.add_argument('--resume_head_path', default='', type=str, help='path of previously trained model head')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--save_folder', type=str, default='./checkpoints/',
                        help='save frequency')
    parser.add_argument('--model_name',
                        default=model,
                        type=str,
                        help=
                        '(resnet | resnet2p1d | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth',
                        default=model_depth,
                        type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--conv1_t_size',
                        default=conv1_t_size,
                        type=int,
                        help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride',
                        default=conv1_t_stride,
                        type=int,
                        help='Stride in t dim of conv1.')
    parser.add_argument('--no_max_pool',
                        action='store_true',
                        help='If true, the max pooling after conv1 is removed.')
    parser.add_argument('--resnet_shortcut',
                        default=resnet_shortcut,
                        type=str,
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument('--resnet_widen_factor',
                        default=resnet_widen_factor,
                        type=float,
                        help='The number of feature maps of resnet is multiplied by this value')
    parser.add_argument('--n_classes',
                        default=n_classes,
                        type=int,
                        help='Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_pretrain_classes',
                        default=n_pretrain_classes,
                        type=int,
                        help=('Number of classes of pretraining task.'
                            'When using --pretrain_path, this must be set.'))
    parser.add_argument('--n_input_channels',
                        default=n_input_channels,
                        type=int,
                        help=('Number of input_channels'))
    parser.add_argument('--pretrain_path',
                        default=pretrain_path,
                        type=Path,
                        help='Pretrained model path (.pth).')
    parser.add_argument('--accimage',
                        action='store_true',
                        help='If true, accimage is used to load images.')
    parser.add_argument('--wide_resnet_k',
                        default=wide_resnet_k,
                        type=int,
                        help='Wide resnet k')
    parser.add_argument('--resnext_cardinality',
                        default=resnext_cardinality,
                        type=int,
                        help='ResNeXt cardinality')
    parser.add_argument('--input_type',
                        default=input_type,
                        type=str,
                        help='(rgb | flow)')
    parser.add_argument('--output_topk',
                        default=output_topk,
                        type=int,
                        help='Top-k scores are saved in json file.')
    #Related Model(Hyperparameter)
    parser.add_argument('--view', default='Right', type=str, help='Dashboard | Rear | Right')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2, help='learning rate') #0.001 ->0.2
    parser.add_argument('--lr_decay_epochs', type=str, default='150,200,250',  
                        help='where to decay lr, can be a list')
    #parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--epochs', default=250, type=int, help='Number of total epochs to run')
    parser.add_argument('--norm_value', default=255, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--cal_vec_batch_size', default=64, type=int,
                        help='batch size for calculating normal driving average vector.')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='a temperature parameter that controls the concentration level of the distribution of embedded vectors')
    parser.add_argument('--manual_seed', default=manual_seed, type=int, help='Manually set random seed')
    parser.add_argument('--memory_bank_size', default=200, type=int, help='Memory bank size')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--lr_decay', default=100, type=int,
                        help='Number of epochs after which learning rate will be reduced to 1/10 of original value')
    parser.add_argument('--initial_scales', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--scale_step', default=0.9, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--n_scales', default=3, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--train_crop', default='random', type=str,
                        help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.\
                        (random | corner | center)')
    parser.add_argument('--a_split_ratio', default=1.0, type=float,
                        help='the ratio of normal driving samples will be used during training')
    parser.add_argument('--n_split_ratio', default=1.0, type=float,
                        help='the ratio of normal driving samples will be used during training')
    parser.add_argument('--Z_momentum', default=0.9, help='momentum for normalization constant Z updates')
    parser.add_argument('--downsample', default=2, type=int, help='Downsampling. Select 1 frame out of N')
    parser.add_argument('--window_size', default=6, type=int, help='the window size for post-processing')
    #GPU
    parser.add_argument('--use_cuda', default=True, type=ast.literal_eval, help='If true, cuda is used.')
    parser.add_argument('--n_threads', default=2, type=int, help='num of workers loading dataset')
    #Related Distribution 
    parser.add_argument('--tracking', default=True, type=ast.literal_eval,
                        help='If true, BN uses tracking running stats')
    parser.add_argument('--dist_rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--distributed',
                        default=False,
                        action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs.')
    parser.add_argument('--world_size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist_url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    arg=parse_args()
    print(arg)