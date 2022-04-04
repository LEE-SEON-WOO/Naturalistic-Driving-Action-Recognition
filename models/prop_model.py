from email import iterators
import numpy as np 
import torch.nn as nn
import torch
import torch.nn.functional as F
from models import backbone
"""
*r2p1d50_K_200ep.pth --model resnet --model_depth 50 --n_pretrain_classes 700 -> 93.4 (1st)
*r3d50_KM_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 1039 -> 92.9 (2nd)
r3d50_K_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 700 ->  92.0 (3rd)
r3d34_K_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 700 -> 88.8 (4th)
r3d18_K_200ep.pth: --model resnet --model_depth 18 --n_pretrain_classes 700 -> 87.8 (5th)
"""

import headers
headers.all_from('py')
from headers.head import ProjectionHead

class R3D_MLP(nn.Module):
    def __init__(self, feature_dim, model_depth, opt):
        super(R3D_MLP, self).__init__()
        self.proj_head = ProjectionHead(output_dim=feature_dim, 
                                        model_depth=model_depth)
        
        print("Projection Head Maked!")
        
        self.r3d = generate_model(opt)
        print("ResNet C3D Maked!")
        
    def forward(self, x):
        unnormed_x = self.r3d(x)
        normed_x = F.normalize(unnormed_x, p=2, dim=1)
        return self.proj_head(unnormed_x), normed_x

def generate_model(opt):
    model = build_backbone(opt)
    
    model = load_pretrained_model(model, 
                                pretrain_path=opt.pretrain_path, 
                                model_name=opt.model_name)
    
    if torch.cuda.is_available():
        model = model.cuda()
    return model
from collections import OrderedDict
def load_pretrained_model(model, pretrain_path, model_name):
    if str(pretrain_path).endswith('pth'):
        print('loading pretrained model {}'.format(pretrain_path))
        #Added classifier###########################################
        """
        tmp_model = model
        if model_name == 'densenet': 
            tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                            n_finetune_classes)
        else:
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                    n_finetune_classes)
        """
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrain_path)['state_dict']
        pretrained_dict = OrderedDict()
        
        for name, module in model.named_children():
            if name in model_dict:
                
                pretrained_dict[name]=module
            
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        #Modified First Conv Layer
        model = _construct_depth_model(model)
        
    return model

def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]

def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append({'params': v})

    return parameters

def build_backbone(opt):
    from backbone import resnet, resnet2p1d, wide_resnet, resnext, densenet
    if str(opt.model_name) == 'resnet':
        model = resnet.generate_model(model_depth=opt.model_depth,
                                    n_classes=opt.n_classes,
                                    n_input_channels=opt.n_input_channels,
                                    shortcut_type=opt.resnet_shortcut,
                                    conv1_t_size=opt.conv1_t_size,
                                    conv1_t_stride=opt.conv1_t_stride,
                                    no_max_pool=opt.no_max_pool,
                                    widen_factor=opt.resnet_widen_factor)
                                    
    elif str(opt.model_name) == 'resnet2p1d':
        model = resnet2p1d.generate_model(model_depth=opt.model_depth,
                                        n_classes=opt.n_classes,
                                        n_input_channels=opt.n_input_channels,
                                        shortcut_type=opt.resnet_shortcut,
                                        conv1_t_size=opt.conv1_t_size,
                                        conv1_t_stride=opt.conv1_t_stride,
                                        no_max_pool=opt.no_max_pool,
                                        widen_factor=opt.resnet_widen_factor)
    elif str(opt.model_name) == 'wideresnet':
        model = wide_resnet.generate_model(model_depth=opt.model_depth,
                                        k=opt.wide_resnet_k,
                                        n_classes=opt.n_classes,
                                        n_input_channels=opt.n_input_channels,
                                        shortcut_type=opt.resnet_shortcut,
                                        conv1_t_size=opt.conv1_t_size,
                                        conv1_t_stride=opt.conv1_t_stride,
                                        no_max_pool=opt.no_max_pool)
    elif str(opt.model_name) == 'resnext':
        model = resnext.generate_model(model_depth=opt.model_depth,
                                    cardinality=opt.resnext_cardinality,
                                    n_classes=opt.n_classes,
                                    n_input_channels=opt.n_input_channels,
                                    shortcut_type=opt.resnet_shortcut,
                                    conv1_t_size=opt.conv1_t_size,
                                    conv1_t_stride=opt.conv1_t_stride,
                                    no_max_pool=opt.no_max_pool)
    elif str(opt.model_name) == 'densenet':
        model = densenet.generate_model(model_depth=opt.model_depth,
                                        n_classes=opt.n_classes,
                                        n_input_channels=opt.n_input_channels,
                                        conv1_t_size=opt.conv1_t_size,
                                        conv1_t_stride=opt.conv1_t_stride,
                                        no_max_pool=opt.no_max_pool)

    return model

def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model

def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device == 'cuda' and device is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device == 'cuda':
        print("Gpu Setted!")
        model = nn.DataParallel(model, device_ids=None).cuda()
    return model

def _construct_depth_model(base_model):
    # modify the first convolution kernels for Depth input
    modules = list(base_model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                 list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]
    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (1*motion_length,  ) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
    new_conv = nn.Conv3d(1, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding, bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name
    # replace the first convlution layer
    setattr(container, layer_name, new_conv)
    return base_model

"""
import argparse
from pathlib import Path


def parse_opts(pretrain_path:str='../pretrained/r3d50_KMS_200ep.pth', 
                model:str='resnet',
                n_input_channels:int=3,
                model_depth:int=50,
                manual_seed:int=1,
                output_topk:int=5,
                input_type:str='rgb',
                n_classes:int=1139,
                n_pretrain_classes:int=1139,
                conv1_t_size:int=7,
                conv1_t_stride:int=1,
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
"""



if __name__ == '__main__':
    # summary(model, (16, 3, 112, 112))
    input = torch.rand(5, 3, 16, 112, 112).cuda()
    #r2p1d50_K_200ep.pth --model resnet --model_depth 50 --n_pretrain_classes 700 -> 93.4 (1st)
    saved_model_file = '../pretrained/r3d50_KM_200ep.pth'
    opt = parse_opts()
    
    #model = generate_model(opt, removed_classifier=True)


    model = R3D_MLP(128, 50, opt).cuda()
    inp = torch.rand(8, 3, 16, 112, 112).cuda()
    print(model(inp)[0].shape[0], model(inp)[0].shape[1])