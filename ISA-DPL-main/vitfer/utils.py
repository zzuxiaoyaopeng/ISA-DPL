import torch
import numpy as np
import os
import math
from sklearn.metrics import f1_score
from math import cos, pi


# 使用：输入batch图片及对应的标签,返回混合图像mixed_x以及标签y_a,y_b
# 混合损失: mix_loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
def mixup_data(x, y, alpha=1.0):
    # 随机生成一个beta分布的参数lam,用于生成随机的线性组合,以实现mixup数据扩充
    lam = np.random.beta(alpha, alpha)
    #生 成一个随机的序列,用于将输入数据进行shuffle
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    # 得到混合后的新图片
    mixed_x = lam * x + (1 - lam) * x[index, :]
    # 得到混图对应的两类标签
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

# 计算准确度
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # 不连续tensor使用view报错误:at leat one dimension spans across two contiguous subspaces
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

# f1 score精确率和召回率的调和平均数
def EXPR_metric(x, y):
    # x: predict; y: target
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)

    f1 = f1_score(x, y, average= 'macro')
    acc = sum(x==y) / x.shape[0]

    return f1, acc, 0.67 * f1 + 0.33 * acc

# rampup_length由epochs赋值
def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def save_best_checkpoint(state, is_best):
    full_bestname = os.path.join('./trained_weights', 'model_best.pth')
    if is_best:
        torch.save(state, full_bestname)

def save_last_checkpoint(state):
    last_name = os.path.join('./trained_weights', 'model_last.pth')
    torch.save(state, last_name)


# warm up
def adjust_learning_rate(optimizer, current_epoch, max_epoch=20, lr_min=0, lr_max=1e-4):
    warmup_epoch = 10
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    elif current_epoch < max_epoch:
        lr = lr_min + (lr_max - lr_min) * (1 + cos(pi*(current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    else:
        lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * (current_epoch - max_epoch) / (max_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 假设batch_size为2,此时max_probs为tensor([0.1454,0.1455])
# 注意max_idx为无标签数据预测得到的伪标签
# threshold为[0.8,0.8,0.8,0.8,0.8,0.8,0.8]
def mask_generate(max_probs, max_idx, batch, threshold):
    # mask_ori为tensor([0,0...])
    mask_ori = torch.zeros(batch)
    mask_ori = mask_ori.cuda()
    for i in range(7):
        # idx为numpy.ndarray,存放max_idx中元素的索引值
        # 比如某个特征图最大预测值为0.1455,对应标签为3,标签所在索引为1
        idx = np.where(max_idx.cpu() == i)[0]
        # 进行无标签数据集划分
        # max_probs[idx]取出对应位置的最大预测值,如果大于置信度阈值边界,m设置为1,否则为0
        # m为列表,存放0或1,代表当前类别i对应的数据集被划分为高置信度数据集以及低置信度数据集
        m = max_probs[idx].ge(threshold[i]).float()
        for k in range(len(idx)):
            # idx[k]取出的都是max_idx中元素的索引位置
            # mask_ori长度与max_idx长度一致,长度为batch_size
            # 将符合条件的mask_ori对应位置元素改为1
            # 即代表该batch个无标签数据集中,mask_ori元素为1的样本是高置信度数据
            mask_ori[idx[k]] += m[k]

    return mask_ori.cuda()


# 置信度边界自适应改变算法
# threshold初始代表每个类别的置信度边界值,即0.8
def adaptive_threshold_generate(outputs, targets, threshold, epoch):
    # 每张训练图片的预测得分
    probs = torch.softmax(outputs, dim=1)
    # argmax进行预测最大得分所在索引
    max_probs, max_idx = torch.max(probs, dim=1)
    # 如果预测索引位置与真实标签targets相同,则认为是正确的样本
    eq_idx = np.where(targets.eq(max_idx).cpu() == 1)[0]

    # 获取正确样本预测的得分与标签值
    probs_new = max_probs[eq_idx]
    targets_new = targets[eq_idx]
    # 类别为c的置信度边界公式Tc:类别c中所有正确预测样本的得分乘以对应预测标签值的总和除以类别为c的正确预测的样本数量
    # 根据上述公式Tc然后加上变量epoch,构造随时间变化的置信度自适应公式
    for i in range(7):
        idx = np.where(targets_new.cpu() == i)[0]
        if idx.shape[0] != 0:
            # Tc*B/(pure_embedding+γ-t)
            threshold[i] = probs_new[idx].mean().cpu() * 0.97 / (1 + math.exp(-1 * epoch)) \
                if probs_new[idx].mean().cpu() * 0.97 / (1 + math.exp(-1 * epoch)) >= 0.8 else 0.8
        else:
            threshold[i] = 0.8

    return threshold
