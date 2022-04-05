import argparse
import ast

def parse_args():
    parser = argparse.ArgumentParser(description='DAD training on Videos')
    parser.add_argument('--root_path', default='../A1/newFrame/', type=str, help='root path of the dataset')
    parser.add_argument('--mode', default='train', type=str, help='train | test(validation)')
    parser.add_argument('--view', default='Right', type=str, help='Dashboard | Rear | Right')
    parser.add_argument('--feature_dim', default=128, type=int, help='To which dimension will video clip be embedded')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of each video clip')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--model_type', default='resnet', type=str, help='so far only resnet')
    parser.add_argument('--model_depth', default=50, type=int, help='Depth of resnet (18 | 50 | 101)')
    parser.add_argument('--shortcut_type', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--pre_train_model', default=True, type=ast.literal_eval, help='Whether use pre-trained model')
    parser.add_argument('--use_cuda', default=True, type=ast.literal_eval, help='If true, cuda is used.')
    parser.add_argument('--n_train_batch_size', default=3, type=int, help='Batch Size for normal training data')
    parser.add_argument('--a_train_batch_size', default=25, type=int, help='Batch Size for anormal training data')
    parser.add_argument('--val_batch_size', default=25, type=int, help='Batch Size for validation data')
    parser.add_argument('--dist_rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')
    
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--epochs', default=250, type=int, help='Number of total epochs to run')
    parser.add_argument('--n_threads', default=2, type=int, help='num of workers loading dataset')
    parser.add_argument('--tracking', default=True, type=ast.literal_eval,
                        help='If true, BN uses tracking running stats')
    parser.add_argument('--norm_value', default=255, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--cal_vec_batch_size', default=64, type=int,
                        help='batch size for calculating normal driving average vector.')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='a temperature parameter that controls the concentration level of the distribution of embedded vectors')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--memory_bank_size', default=200, type=int, help='Memory bank size')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--lr_decay', default=100, type=int,
                        help='Number of epochs after which learning rate will be reduced to 1/10 of original value')
    parser.add_argument('--distributed',
                        default=False,
                        action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs.')
    parser.add_argument('--world_size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--resume_path', default='', type=str, help='path of previously trained model')
    parser.add_argument('--resume_head_path', default='', type=str, help='path of previously trained model head')
    parser.add_argument('--initial_scales', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--scale_step', default=0.9, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--n_scales', default=3, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--train_crop', default='corner', type=str,
                        help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--checkpoint_folder', default='./checkpoints/', type=str, help='folder to store checkpoints')
    parser.add_argument('--log_folder', default='./logs/', type=str, help='folder to store log files')
    parser.add_argument('--log_resume', default=False, type=ast.literal_eval, help='True|False: a flag controlling whether to create a new log file')
    parser.add_argument('--normvec_folder', default='./normvec/', type=str, help='folder to store norm vectors')
    parser.add_argument('--score_folder', default='./score/', type=str, help='folder to store scores')
    parser.add_argument('--Z_momentum', default=0.9, help='momentum for normalization constant Z updates')
    parser.add_argument('--groups', default=3, type=int, help='hyper-parameters when using shufflenet')
    parser.add_argument('--width_mult', default=2.0, type=float,
                        help='hyper-parameters when using shufflenet|mobilenet')
    parser.add_argument('--val_step', default=1, type=int, help='validate per val_step epochs')
    parser.add_argument('--downsample', default=2, type=int, help='Downsampling. Select 1 frame out of N')
    parser.add_argument('--save_step', default=10, type=int, help='checkpoint will be saved every save_step epochs')
    parser.add_argument('--n_split_ratio', default=1.0, type=float,
                        help='the ratio of normal driving samples will be used during training')
    parser.add_argument('--a_split_ratio', default=1.0, type=float,
                        help='the ratio of normal driving samples will be used during training')
    parser.add_argument('--window_size', default=6, type=int, help='the window size for post-processing')
    parser.add_argument('--dist_url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    arg=parse_args()
    print(arg)