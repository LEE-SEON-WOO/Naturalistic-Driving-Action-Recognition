from __future__ import print_function
from torch import nn
import torch
import math
import numpy as np
import torch.optim as optim
import csv
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
class Logger(object):
    """Logger object for training process, supporting resume training"""
    def __init__(self, path, header, resume=False):
        """
        :param path: logging file path
        :param header: a list of tags for values to track
        :param resume: a flag controling whether to create a new
        file or continue recording after the latest step
        """
        self.log_file = None
        self.resume = resume
        self.header = header
        if not self.resume:
            self.log_file = open(path, 'w')
            self.logger = csv.writer(self.log_file, delimiter='\t')
            self.logger.writerow(self.header)
        else:
            self.log_file = open(path, 'a+')
            self.log_file.seek(0, os.SEEK_SET)
            reader = csv.reader(self.log_file, delimiter='\t')
            self.header = next(reader)
            # move back to the end of file
            self.log_file.seek(0, os.SEEK_END)
            self.logger = csv.writer(self.log_file, delimiter='\t')

    def __del__(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for tag in self.header:
            assert tag in values, 'Please give the right value as defined'
            write_values.append(values[tag])
        self.logger.writerow(write_values)
        self.log_file.flush()


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))

def adjust_learning_rate(optimizer, lr_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_rate



def get_fusion_label(csv_path):
    """
    Read the csv file and return labels
    :param csv_path: path of csv file
    :return: ground truth labels
    """
    gt = np.zeros(360000)
    base = -10000
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[-1] == '':
                continue
            if row[1] != '':
                base += 10000
            if row[4] == 'N':
                gt[base + int(row[2]):base + int(row[3]) + 1] = 1
            else:
                continue
    return gt

def evaluate(score, label, whether_plot):
    """
    Compute Accuracy as well as AUC by evaluating the scores
    :param score: scores of each frame in videos which are computed as the cosine similarity between encoded test vector and mean vector of normal driving
    :param label: ground truth
    :param whether_plot: whether plot the AUC curve
    :return: best accuracy, corresponding threshold, AUC
    """
    thresholds = np.arange(0., 1., 0.01)
    best_acc = 0.
    best_threshold = 0.
    for threshold in thresholds:
        prediction = score >= threshold # TODO! 
        correct = prediction[prediction == label]

        acc = (np.sum(correct) / correct.shape[0] * 100)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    AUC = auc(fpr, tpr)

    if whether_plot:
        plt.plot(fpr, tpr, color='r')
        #plt.fill_between(fpr, tpr, color='r', y2=0, alpha=0.3)
        plt.plot(np.array([0., 1.]), np.array([0., 1.]), color='b', linestyle='dashed')
        plt.tick_params(labelsize=23)
        #plt.text(0.9, 0.1, f'AUC: {round(AUC, 4)}', fontsize=25)
        plt.xlabel('False Positive Rate', fontsize=25)
        plt.ylabel('True Positive Rate', fontsize=25)
        plt.show()
    return best_acc, best_threshold, AUC


def post_process(score, window_size=6):
    """
    post process the score
    :param score: scores of each frame in videos
    :param window_size: window size
    :param momentum: momentum factor
    :return: post processed score
    """
    processed_score = np.zeros(score.shape)
    for i in range(0, len(score)):
        processed_score[i] = np.mean(score[max(0, i-window_size+1):i+1])

    return processed_score

def get_normal_vector(model, train_normal_loader_for_test, cal_vec_batch_size, feature_dim, use_cuda):
    total_batch = int(len(train_normal_loader_for_test))
    print("=====================================Calculating Average Normal Vector=====================================")
    if use_cuda:
        normal_vec = torch.zeros((1, 512)).cuda()
    else:
        normal_vec = torch.zeros((1, 512))
    for batch, (normal_data, idx) in enumerate(train_normal_loader_for_test):
        if use_cuda:
            normal_data = normal_data.cuda()
        _, outputs = model(normal_data)
        outputs = outputs.detach()
        normal_vec = (torch.sum(outputs, dim=0) + normal_vec * batch * cal_vec_batch_size) / (
                (batch + 1) * cal_vec_batch_size)
        print(f'\r Calculating Average Normal Vector: Batch {batch + 1} / {total_batch}', end='')
    normal_vec = l2_normalize(normal_vec)
    return normal_vec

def cal_score(model_dashboard, model_rear, model_right, normal_vec_dashboard, normal_vec_rear,
              normal_vec_right, test_loader_dashboard, test_loader_rear, test_loader_right, score_folder, use_cuda):
    """
    Generate and save scores of Dashboard/Right/Rear views
    """
    assert int(len(test_loader_dashboard)) == int(len(test_loader_rear)) == int(len(test_loader_right))
    total_batch = int(len(test_loader_dashboard))
    sim_list = torch.zeros(0)
    sim_1_list = torch.zeros(0)
    sim_2_list = torch.zeros(0)
    sim_3_list = torch.zeros(0)
    # sim_4_list = torch.zeros(0)
    label_list = torch.zeros(0).type(torch.LongTensor)
    for batch, (data1, data2, data3) in enumerate(
            zip(test_loader_dashboard, test_loader_rear, test_loader_right)):
        if use_cuda:
            data1[0] = data1[0].cuda()
            data1[1] = data1[1].cuda()
            data2[0] = data2[0].cuda()
            data2[1] = data2[1].cuda()
            data3[0] = data3[0].cuda()
            data3[1] = data3[1].cuda()


        assert torch.sum(data1[1] == data2[1]) == torch.sum(data2[1] == data3[1]) == data1[1].size(0)

        out_1 = model_dashboard(data1[0])[1].detach()
        out_2 = model_rear(data2[0])[1].detach()
        out_3 = model_right(data3[0])[1].detach()
        

        sim_1 = torch.mm(out_1, normal_vec_dashboard.t())
        sim_2 = torch.mm(out_2, normal_vec_rear.t())
        sim_3 = torch.mm(out_3, normal_vec_right.t())
        
        sim = (sim_1 + sim_2 + sim_3 ) / 4

        sim_list = torch.cat((sim_list, sim.squeeze().cpu()))
        label_list = torch.cat((label_list, data1[1].squeeze().cpu()))
        sim_1_list = torch.cat((sim_1_list, sim_1.squeeze().cpu()))
        sim_2_list = torch.cat((sim_2_list, sim_2.squeeze().cpu()))
        sim_3_list = torch.cat((sim_3_list, sim_3.squeeze().cpu()))
        print(f'/r valuating: Batch {batch + 1} / {total_batch}', end='')

    np.save(os.path.join(score_folder, 'score_dashboard.npy'), sim_1_list.numpy())
    print('score_dashboard.npy is saved')
    np.save(os.path.join(score_folder, 'score_rear.npy'), sim_2_list.numpy())
    print('score_rear.npy is saved')
    np.save(os.path.join(score_folder, 'score_right.npy'), sim_3_list.numpy())
    print('score_right.npy is saved')


def get_score(score_folder, mode):
    """
    !!!Be used only when scores exist!!!
    Get the corresponding scores according to requiements
    :param score_folder: the folder where the scores are saved
    :param mode: Dashboard | Rear | Right | Rear | fusion_Dashboard | fusion_Rear | fusion_Right | fusion_all
    :return: the corresponding scores according to requirements
    """
    if mode not in ['Dashboard', 'Right', 'Rear', 'fusion_DashBoard_Right', 'fusion_DashBoard_Rear', 'fusion_Rear_Right', 'fusion_all']:
        print('Please enter correct mode: Dashboard | Right | Rear | fusion_Dashboard | fusion_Right | fusion_Rear | fusion_all')
        return
    if mode == 'Dashboard':
        score = np.load(os.path.join(score_folder + '/score_dashboard.npy'))
    elif mode == 'Rear':
        score = np.load(os.path.join(score_folder + '/score_rear.npy'))
    elif mode == 'Right':
        score = np.load(os.path.join(score_folder + '/score_Right.npy'))
    elif mode == 'fusion_Dashboard_Right':
        score1 = np.load(os.path.join(score_folder + '/score_Dashboard.npy'))
        score2 = np.load(os.path.join(score_folder + '/score_Right.npy'))
        score = np.mean((score1, score2), axis = 0)
    elif mode == 'fusion_Dashboard_Rear':
        score3 = np.load(os.path.join(score_folder + '/score_Dashboard.npy'))
        score4 = np.load(os.path.join(score_folder + '/score_Rear.npy'))
        score = np.mean((score3, score4), axis=0)
    elif mode == 'fusion_Rear_Right':
        score1 = np.load(os.path.join(score_folder + '/score_Right.npy'))
        score3 = np.load(os.path.join(score_folder + '/score_Rear.npy'))
        score = np.mean((score1, score3), axis=0)
    elif mode == 'fusion_all':
        score1 = np.load(os.path.join(score_folder + '/score_Dashboard.npy'))
        score2 = np.load(os.path.join(score_folder + '/score_Rear.npy'))
        score3 = np.load(os.path.join(score_folder + '/score_Right.npy'))
        score = np.mean((score1, score2, score3), axis=0)

    return score