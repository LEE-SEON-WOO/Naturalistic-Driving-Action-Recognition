import csv
import random
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def split_acc_diff_threshold(model, normal_vec, test_loader, use_cuda):
    """
    Search the threshold that split the scores the best and calculate the corresponding accuracy
    """
    total_batch = int(len(test_loader))
    print("================================================Evaluating================================================")
    total_n = 0
    total_a = 0
    threshold = np.arange(0., 1., 0.01)
    total_correct_a = np.zeros(threshold.shape[0])
    total_correct_n = np.zeros(threshold.shape[0])
    for batch, batch_data in enumerate(test_loader):
        if use_cuda:
            batch_data[0] = batch_data[0].cuda()
            batch_data[1] = batch_data[1].cuda()
        n_num = torch.sum(batch_data[1]).cpu().detach().numpy()
        total_n += n_num
        total_a += (batch_data[0].size(0) - n_num)
        _, outputs = model(batch_data[0])
        outputs = outputs.detach()
        
        similarity = torch.mm(outputs, normal_vec.t())
        for i in range(len(threshold)):
            prediction = similarity >= threshold[i]  # If similarity between sample and average normal vector is smaller than threshold, then this sample is predicted as anormal driving which is set to 0
            correct = prediction.squeeze() == batch_data[1]
            total_correct_a[i] += torch.sum(correct[~batch_data[1].bool()])
            total_correct_n[i] += torch.sum(correct[batch_data[1].bool()])
        print(f'\r Evaluating: Batch {batch + 1} / {total_batch}', end='')

    acc_n = [(correct_n / total_n) for correct_n in total_correct_n]
    acc_a = [(correct_a / total_a) for correct_a in total_correct_a]
    acc = [((total_correct_n[i] + total_correct_a[i]) / (total_n + total_a)) for i in range(len(threshold))]
    best_acc = np.max(acc)
    idx = np.argmax(acc)
    best_threshold = idx * 0.01
    return best_acc, best_threshold, acc_n[idx], acc_a[idx], acc, acc_n, acc_a

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def calculate_precision_and_recall(outputs, targets, pos_label=1):
    with torch.no_grad():
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        precision, recall, _, _ = precision_recall_fscore_support(
            targets.view(-1, 1).cpu().numpy(),
            pred.cpu().numpy())

        return precision[pos_label], recall[pos_label]


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)
