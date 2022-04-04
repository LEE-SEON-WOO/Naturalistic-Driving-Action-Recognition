import argparse
from pathlib import Path


def parse_opts(pretrain_path:str='../pretrained/r3d18_K_200ep.pth', 
                model:str='resnet',
                n_input_channels:int=1,
                model_depth:int=50,
                manual_seed:int=1,
                output_topk:int=5,
                input_type:str='rgb',
                n_classes:int=1139,
                n_pretrain_classes:int=1139,
                conv1_t_size:int=1,
                conv1_t_stride:int=3,
                resnet_shortcut:str='B',
                ft_begin_module:str='',
                batchnorm_sync:bool=True,
                resnet_widen_factor:float=1.0,
                wide_resnet_k:int=2,
                resnext_cardinality:int=32,
                sample_size:int=112,
                sample_duration:int=16,
                ):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_classes',
        default=n_classes,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)')
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
    parser.add_argument('--ft_begin_module',
                        default=ft_begin_module,
                        type=str,
                        help=('Module name of beginning of fine-tuning'
                            '(conv1, layer1, fc, denseblock1, classifier, ...).'
                            'The default means all layers are fine-tuned.'))
    parser.add_argument('--batchnorm_sync',
                        default=batchnorm_sync,
                        type=bool,
                        help='If true, SyncBatchNorm is used instead of BatchNorm.')
    parser.add_argument('--resume_path',
                        default=None,
                        type=Path,
                        help='Save data (.pth) of previous training')
    
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
    parser.add_argument('--manual_seed',
                        default=manual_seed,
                        type=int,
                        help='Manually set random seed')
    parser.add_argument('--accimage',
                        action='store_true',
                        help='If true, accimage is used to load images.')
    parser.add_argument('--output_topk',
                        default=output_topk,
                        type=int,
                        help='Top-k scores are saved in json file.')
    parser.add_argument('--distributed',
                        action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs.')
    args = parser.parse_args()

    return args
