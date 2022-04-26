from utils.temporal_transforms import TemporalSequentialCrop
from utils.temporal_transforms import TemporalCasCadeSampling
from utils.group import OneOf
from utils import spatial_transforms
from data.train_dataset import DAC
from torch.utils.data import DataLoader, Subset
import numpy as np
from loss.NCEAverage import NCEAverage
from loss.NCECriterion import NCECriterion
import os, torch
import torch.backends.cudnn as cudnn
from utils.util import adjust_learning_rate, Logger, l2_normalize
from utils.utils import AverageMeter, split_acc_diff_threshold
from models.prop_model import R3D_MLP, make_data_parallel
from tqdm import tqdm
from models.prop_model import make_data_parallel

def train_epoch(train_normal_loader, train_anormal_loader, model, nce_average, criterion, optimizer, epoch, args,
          batch_logger, epoch_logger, memory_bank=None):
    losses = AverageMeter()
    prob_meter = AverageMeter()

    model.train()
    
    for batch, ((normal_data, idx_n), (anormal_data, idx_a)) in tqdm(enumerate(zip(train_normal_loader, train_anormal_loader))):
        if normal_data.size(0) != args.n_train_batch_size:
            break
        data = torch.cat((normal_data, anormal_data), dim=0)  # n_vec as well as a_vec are all normalized value
        
        if args.use_cuda:
            data = data.cuda()
            idx_a = idx_a.cuda()
            idx_n = idx_n.cuda()
            normal_data = normal_data.cuda()
        
        # ================forward====================
        
        vec, normed_vec = model(data)
        
        n_vec = vec[0:args.n_train_batch_size]
        a_vec = vec[args.n_train_batch_size:]
        outs, probs = nce_average(n_vec, a_vec, idx_n, idx_a, normed_vec[0:args.n_train_batch_size])
        loss = criterion(outs)

        # ================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===========update memory bank===============
        model.eval()
        _, n = model(normal_data)
        n = n.detach()
        average = torch.mean(n, dim=0, keepdim=True)
        if len(memory_bank) < args.memory_bank_size:
            memory_bank.append(average)
        else:
            memory_bank.pop(0)
            memory_bank.append(average)
        model.train()

        # ===============update meters ===============
        losses.update(loss.item(), outs.size(0))
        prob_meter.update(probs.item(), outs.size(0))

        # =================logging=====================
        batch_logger.log({
            'epoch': epoch,
            'batch': batch,
            'loss': losses.val,
            'probs': prob_meter.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        print(
            f'\r Training Process is running: {epoch}/{args.epochs}  | Batch: {batch} | Loss: {losses.val:.3f} ({losses.avg:.3f}) | Probs: {prob_meter.val:.3f} ({prob_meter.avg:.3f},)',end='')
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'probs': prob_meter.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    return memory_bank, losses.avg


def train(args):
    args.scales = [args.initial_scales] # 1.0
    for i in range(1, args.n_scales): # 3
        args.scales.append(args.scales[-1] * args.scale_step) # 1 * 0.9
    
    
    before_crop_duration = int(args.sample_duration * args.downsample) #sample_duration=16, downsample=2
    temporal_transform = OneOf([TemporalSequentialCrop(duration = before_crop_duration, 
                                                downsample = args.downsample),
                                TemporalCasCadeSampling(duration=before_crop_duration,
                                                        downsample = args.downsample) #Our Contribution!!
                                ])#각 데이터에 맞는 transform 시행 -> 우리는 if문 없애고 rgb에 맞게끔
    
    
    spatial_transform = spatial_transforms.Compose([
            spatial_transforms.Scale(args.sample_size),
            spatial_transforms.ToTensor(args.norm_value, args.mode),
            spatial_transforms.Normalize(mean=[0.3685, 0.3685, 0.3683],  #Right
                                        std=[0.2700, 0.2700, 0.2699])])
    print("=================================Loading Driving Training Data!=================================")
    training_anormal_data = DAC(root_path=args.root_path,
                                subset='train',
                                view=args.view,
                                sample_duration=before_crop_duration,
                                type='anormal',
                                spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform)

    training_anormal_size = int(len(training_anormal_data) * args.a_split_ratio)
    training_anormal_data = Subset(training_anormal_data, np.arange(training_anormal_size))
    
    train_anormal_loader = DataLoader(
        training_anormal_data,
        batch_size=args.a_train_batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=False,
    )

    print("=================================Loading Normal-Driving Training Data!=================================")
    training_normal_data = DAC(root_path=args.root_path,
                                subset='train',
                                view=args.view,
                                sample_duration=before_crop_duration,
                                type='normal',
                                spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform
                                )

    training_normal_size = int(len(training_normal_data) * args.n_split_ratio)
    training_normal_data = Subset(training_normal_data, np.arange(training_normal_size))

    train_normal_loader = DataLoader(
        training_normal_data,
        batch_size=args.n_train_batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=False,
    )

    print("========================================Loading Validation Data========================================")
    val_spatial_transform = spatial_transforms.Compose([
        spatial_transforms.Scale(args.sample_size),
        spatial_transforms.ToTensor(args.norm_value, 
                                    args.mode),
        spatial_transforms.Normalize([0], [1])
    ])
    validation_data = DAC(root_path=args.root_path,
                        subset='validation',
                        view=args.view,
                        sample_duration=args.sample_duration,
                        type=None,
                        spatial_transform=val_spatial_transform,
                        )
    
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=False,
    )

    len_neg = training_anormal_data.__len__()
    len_pos = training_normal_data.__len__()
    num_val_data = validation_data.__len__()
    print(f'len_neg: {len_neg}')
    print(f'len_pos: {len_pos}')
    print(f'val_data len:{num_val_data}')
    import copy
    print("============================================Generating Model============================================")
    if args.resume_path == '':
        # ===============generate new model or pre-trained model===============
        opt = copy.deepcopy(args)
        
        model = R3D_MLP(feature_dim=args.feature_dim,model_depth=args.model_depth, opt=opt)
        model = make_data_parallel(model, opt.distributed, device='cuda')
        
        optimizer = torch.optim.SGD(list(model.parameters()), lr=args.learning_rate, momentum=args.momentum,
                                    dampening=float(args.dampening), weight_decay=args.weight_decay, nesterov=args.nesterov)
        nce_average = NCEAverage(args.feature_dim, len_neg, len_pos, args.tau, args.Z_momentum)
        criterion = NCECriterion(len_neg)
        begin_epoch = 1
        best_acc = 0
        memory_bank = []
    else:
        # ===============load previously trained model ===============
        args.pre_train_model = False
        
        model = R3D_MLP(feature_dim=args.feature_dim,model_depth=args.model_depth, opt=opt)
        
        
        #model.load_state_dict(resume_checkpoint['state_dict'])
        
        optimizer = torch.optim.SGD(list(model.parameters()), lr=args.learning_rate, momentum=args.momentum,
                                    dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)
        #optimizer.load_state_dict(resume_checkpoint['optimizer'])
        nce_average = resume_checkpoint['nce_average']
        criterion = NCECriterion(len_neg) #
        begin_epoch = resume_checkpoint['epoch'] + 1
        best_acc = resume_checkpoint['acc']
        memory_bank = resume_checkpoint['memory_bank']
        del resume_checkpoint
        torch.cuda.empty_cache()
        adjust_learning_rate(optimizer, args.learning_rate)

    print("==========================================!!!START TRAINING!!!==========================================")
    cudnn.benchmark = True
    batch_logger = Logger(os.path.join(args.log_folder, 'batch.log'), ['epoch', 'batch', 'loss', 'probs', 'lr'], args.log_resume)
    epoch_logger = Logger(os.path.join(args.log_folder, 'epoch.log'), ['epoch', 'loss', 'probs', 'lr'], args.log_resume)
    val_logger = Logger(os.path.join(args.log_folder, 'val.log'),
                        ['epoch', 'accuracy', 'normal_acc', 'anormal_acc', 'threshold', 'acc_list',
                        'normal_acc_list', 'anormal_acc_list'], args.log_resume)

    for epoch in range(begin_epoch, begin_epoch + args.epochs + 1):
        memory_bank, loss = train_epoch(train_normal_loader=train_normal_loader, 
                                        train_anormal_loader=train_anormal_loader, 
                                        model=model, 
                                        nce_average=nce_average,
                                        criterion=criterion, 
                                        optimizer=optimizer, 
                                        epoch=epoch, args=args, batch_logger=batch_logger, 
                                        epoch_logger=epoch_logger, 
                                        memory_bank=memory_bank)

        if epoch % args.val_step == 0:
            print("==========================================!!!Validation!!!==========================================")
            normal_vec = torch.mean(torch.cat(memory_bank, dim=0), dim=0, keepdim=True)
            normal_vec = l2_normalize(normal_vec)

            model.eval()
            accuracy, best_threshold, acc_n, acc_a, acc_list, acc_n_list, acc_a_list = split_acc_diff_threshold(
                model, normal_vec, validation_loader, args.use_cuda)
            print( f'Epoch: {epoch}/{args.epochs} | Accuracy: {accuracy} | Normal Acc: {acc_n} | Anormal Acc: {acc_a} | Threshold: {best_threshold}')
            print(
                "==========================================!!!Logging!!!==========================================")
            val_logger.log({
                'epoch': epoch,
                'accuracy': accuracy * 100,
                'normal_acc': acc_n * 100,
                'anormal_acc': acc_a * 100,
                'threshold': best_threshold,
                'acc_list': acc_list,
                'normal_acc_list': acc_n_list,
                'anormal_acc_list': acc_a_list
            })
            if accuracy > best_acc:
                best_acc = accuracy
                print(
                    "==========================================!!!Saving!!!==========================================")
                checkpoint_path = os.path.join(args.checkpoint_folder, f'best_model_{args.model_type}_{args.view}.pth')
                states = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'acc': accuracy,
                    'threshold': best_threshold,
                    'nce_average': nce_average,
                    'memory_bank': memory_bank}
                torch.save(states, checkpoint_path)

        if epoch % args.save_step == 0:
            print("==========================================!!!Saving!!!==========================================")
            checkpoint_path = os.path.join(args.checkpoint_folder,
                                            f'{args.model_type}_{args.view}_{epoch}.pth')
            states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acc': accuracy,
                'nce_average': nce_average,
                'memory_bank': memory_bank}
            torch.save(states, checkpoint_path)

        if epoch % args.lr_decay == 0:
            lr = args.learning_rate * (0.1 ** (epoch // args.lr_decay))
            adjust_learning_rate(optimizer, lr)
            print(f'New learning rate: {lr}')