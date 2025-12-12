from __future__ import print_function
import math
import sys
import warnings
from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from typing import Optional
from torch.autograd.function import Function


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    # Compute loss for mode
    # If both labels and mask are None,it degenerates to SimCLR unsupervised loss
    def forward(self, features, labels=None, mask=None):
        """
        features: [batch_size,1,512]
        labels: ground truth of shape [batch_size]
        mask: contrastive mask of shape [batch_size,batch_size], mask_{i,j}=pure_embedding if sample j has the same class as sample i
        """
        # 指定GPU或CPU
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        # 维度判断
        if len(features.shape) < 3:
            raise ValueError('features needs to be [bsz, n_views, ...],at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # torch.eq:两个Tensor进行逐元素的比较,若相同位置的两个元素相同,则返回True
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            # labels为None,mask不为None的情况
            mask = mask.float().to(device)

        # 对比分支数,即上下分支
        contrast_count = features.shape[1]
        # torch.unbind(features, dim=pure_embedding): [batch_size,embedding_one_soft,512]->两个[batch_size,512]
        # torch.cat(dim=0): [batch_size*embedding_one_soft,512]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # 默认使用'all'
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # anchor_dot_contrast:[batch_size*embedding_one_soft/temperature,batch_size*embedding_one_soft/temperature]
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # anchor_count与contrast_count值均为2时,代表mask整体对第一维度和第二维度倍乘2
        # 假如mask.shape:[16,16], mask.repeat(embedding_one_soft,embedding_one_soft) -> mask.shape:[32,32]
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask),
                                    1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                                    0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # mean_log_prob_pos = (mask * log_prob).sum(pure_embedding) / mask.sum(pure_embedding + 1e-8)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp

        return loss.mean()


def to_one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)
    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")
    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


# 统一focal_loss与ce_loss
class PolyLoss(_Loss):
    def __init__(self,
                 softmax: bool = False,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 num_class=7
                 ) -> None:
        super().__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')
        self.num_class = num_class

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input: the shape should be BNH[WD], where N is the number of classes.
               You can pass logits or probabilities as input, if pass logit, must set softmax=True
        target: the shape should be BNH[WD] (one-hot format) or B1H[WD], where N is the number of classes.
                It should contain binary values
        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]

        # target not in one-hot encode format, has shape B1H[WD]
        if n_pred_ch != n_target_ch:
            # squeeze out the channel dimension of size 1 to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.squeeze(target, dim=1).long())
            # convert into one-hot format to calculate ce loss
            target = to_one_hot(target, num_classes=n_pred_ch)
        else:
            # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.argmax(target, dim=1))

        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        pt = (input * target).sum(dim=1)  # BH[WD]
        poly_loss = self.ce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            polyl = poly_loss.unsqueeze(1)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return polyl


class PolyCrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int = 7,
                 epsilon: float = 1.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None):  # manual rescaling weight for each class, passed to Cross-Entropy loss
        super(PolyCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    # logits: tensor of shape [N, num_classes]
    # labels: tensor of shape [N]
    def forward(self, logits, labels):
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device, dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits, target=labels, reduction='none', weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1


class PolyFocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int = 7,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None,
                 # manual rescaling weight for each class, passed to binary Cross-Entropy loss
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False):  # set to True if labels are one-hot encoded
        super(PolyFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot

    # logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
    # labels: ground truth tensor of shape [N] or [N, ...]
    def forward(self, logits, labels):
        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=self.num_classes)
            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)

        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL
        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


############################################DANFER######################################################################

class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=7, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.device = device
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)  # [bs,512,1,1]
        batch_size = x.size(0)

        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class PartitionLoss(nn.Module):
    def __init__(self):
        super(PartitionLoss, self).__init__()
        self.eps = sys.float_info.epsilon

    def forward(self, x):
        num_head = x.size(1)
        if num_head > 1:
            var = x.var(dim=1).mean()
            # add eps to avoid empty var case
            loss = torch.log(1 + num_head / (var + self.eps))
        else:
            loss = 0

        return loss


############################################DANFER######################################################################

class FDRL_CompactnessLoss(nn.Module):
    def __init__(self):
        super(FDRL_CompactnessLoss, self).__init__()

    # x:[9,bs,128]
    def forward(self, x):
        bs = x.shape[1]
        M = x.shape[0]

        # 计算每个潜在特征的中心
        c = []
        for i in range(M):
            I = x[i]
            p = torch.sum(I, dim=0)
            p = torch.divide(p, bs)
            c.append(p)
        c = torch.concat(c).reshape(9, 128)

        loss = []
        for i in range(bs):
            for j in range(M):
                out = x[j][i] - c[j]
                # out = torch.norm(out,embedding_one_soft) #L2范数
                out = out ** 2
                loss.append(out)

        total_loss = 0
        for i in range(len(loss)):
            total_loss = total_loss + loss[i]
        last_loss = total_loss / bs

        return last_loss


# 给损失函数添加正则项
class Regularization(nn.Module):
    def __init__(self, model, weight_decay, p=2):
        super(Regularization, self).__init__()
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)

    # 得到正则化参数
    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)

        return weight_list

    # 求取正则化的和
    def regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
        reg_loss = weight_decay * reg_loss

        return reg_loss

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)

        return reg_loss


class Orthognal_Loss(nn.Module):
    def __init__(self):
        super(Orthognal_Loss, self).__init__()

    def forward(self, x, y, z):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        l_12 = torch.sum(x * y, dim=1)
        l_13 = torch.sum(x * z, dim=1)
        l_23 = torch.sum(y * z, dim=1)

        return torch.mean((l_12 + l_13 + l_23) / 3, dim=-1)


###############################################arbex####################################################################

class AnchorLoss(nn.Module):
    def __init__(self, dim_emb=512):
        super().__init__()
        self.factor = math.sqrt(dim_emb)

    def forward(self, anchors):
        n_classes, k, _ = anchors.shape
        anchors = anchors.view(n_classes, -1)
        distances = (anchors.unsqueeze(0) - anchors.unsqueeze(1)) ** 2
        distances = distances / self.factor
        loss = -distances.sum() / k / k

        return loss


class CenterLoss(nn.Module):
    def __init__(self, reduction='mean', dim_emb=512):
        super().__init__()
        self.reduction = reduction
        self.factor = math.sqrt(dim_emb)

    def forward(self, distances, labels, confidence):
        # distances are [batch, n_class, n_anchors]
        distances = distances[range(len(labels)), labels]  # [batch, n_anchors]
        # pick the closest
        distances = torch.min(distances, 1).values  # [batch]
        # loss
        loss = distances * confidence.view(-1) / self.factor
        if self.reduction == 'mean':
            return loss.sum() / len(loss)

        return loss.sum()


class SparseCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(SparseCenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        self.sparse_centerloss = SparseCenterLossFunction.apply
        self.feat_dim = feat_dim
        self.size_average = size_average
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.centers.data.t())

    def forward(self, feat, A, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.sparse_centerloss(feat, A, label, self.centers, batch_size_tensor)

        return loss


class SparseCenterLossFunction(Function):
    @staticmethod
    def forward(ctx, feature, A, label, centers, batch_size):
        ctx.save_for_backward(feature, A, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (A * (feature - centers_batch).pow(2)).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, A, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = feature - centers_batch
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        # A gradient
        grad_A = diff.pow(2) / 2.0 / batch_size

        counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), - A * diff)
        grad_centers = grad_centers / counts.view(-1, 1)

        return grad_output * A * diff / batch_size, grad_output * grad_A, None, grad_centers, None


class DistLoss(nn.Module):
    """
    NLL Loss applied to probability distribution.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.NLLLoss()

    def forward(self, x, l):
        return self.loss(torch.log(x), l)


###############################################arbex####################################################################


########################################################################################################################

# 对每一个样本选择hardest triplet进行训练
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    # [bs, feat_dim],[bs, classes]
    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)

        return self.ranking_loss(dist_an, dist_ap, y)


# triplet学习的是样本间的相对距离,没有学习绝对距离,尽管考虑了类间的离散性,但没有考虑类内的紧凑性
# Center Loss希望可以通过学习每个类的类中心,使得类内的距离变得更加紧凑,但其只约束内内的距离,对内间距离无约束
class CenterLoss_Triplet(nn.Module):
    def __init__(self, num_classes=7, feat_dim=512):
        super(CenterLoss_Triplet, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    # [bs, feat_dim],[bs, classes]
    def forward(self, x, labels):
        batch_size = x.size(0)

        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


###############################################arbex####################################################################


# 似乎没用,损失出现nan,可能少了softmax?
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)

        return loss_sum


# center loss的升级版
# 在关注类别的类内距离的同时,优化类间距离,使得每个类别拥有更大的margin,从而迫使网络能够学习到更具判别性的特征
class IslandLoss(nn.Module):
    def __init__(self, features_dim=512, num_class=7, lamda=1., lamda1=10., scale=1.0, batch_size=32):
        super(IslandLoss, self).__init__()
        self.lamda = lamda
        self.lamda1 = lamda1
        self.num_class = num_class
        self.scale = scale
        self.batch_size = batch_size
        self.feat_dim = features_dim
        self.feature_centers = nn.Parameter(torch.randn([num_class, features_dim]))

    def forward(self, output_features, y_truth):
        """
        output_features: conv层输出的特征,[b,c,h,w]
        y_truth:  标签值  [b,]
        """
        batch_size = y_truth.size(0)
        num_class = self.num_class
        output_features = output_features.view(batch_size, -1)
        assert output_features.size(-1) == self.feat_dim

        factor = self.scale / batch_size
        centers_batch = self.feature_centers.index_select(0, y_truth.long())  # [b,features_dim]
        diff = output_features - centers_batch

        # 1 先求 center loss
        loss_center = 1 / 2.0 * (diff.pow(2).sum()) * factor
        # 2 再求 类心余弦距离
        # 每个类心求余弦距离，+1 使得范围为0-2，越接近0表示类别差异越大，从而优化Loss即使得类间距离变大
        centers = self.feature_centers
        # 求出向量模长矩阵 ||Ci||
        centers_mod = torch.sum(centers * centers, dim=1, keepdim=True).sqrt()  # [num_class, 1]

        #  ====================== method 1 =======================
        item1_sum = 0
        for j in range(num_class):
            dis_sum_j_others = 0
            for k in range(j + 1, num_class):
                dot_kj = torch.sum(centers[j] * centers[k])
                fenmu = centers_mod[j] * centers_mod[k] + 1e-9
                cos_dis = dot_kj / fenmu
                dis_sum_j_others += cos_dis + 1.
            item1_sum += dis_sum_j_others
        loss_island = self.lamda * (loss_center + self.lamda1 * item1_sum)

        # ====================== method 2 =======================
        # # Ci X Ci.T
        # centers_mm = torch.matmul(centers,centers.t())  # [num_class, num_class]
        # centers_mod_mm = centers_mod.mm(centers_mod.t())  # [num_class,num_class]
        # # 求出 cos距离 矩阵, 这是一个对称矩阵
        # centers_cos_dis = centers_mm / centers_mod_mm
        # centers_cos_dis += 1.
        # # 只获取上三角, 代表同一个类别的距离不考虑
        # centers_cos_dis_1 = torch.triu(centers_cos_dis,diagonal=1)
        # print(centers_cos_dis_1)
        # sum_centers_cos_dis = torch.sum(centers_cos_dis_1)
        # loss_island = self.lamda * (loss_center + self.lamda1 * sum_centers_cos_dis)

        return loss_island


# includes the orginal loss (criterion) and a extra distillation loss (criterion)
# computes the loss with different type options, between current model and a teacher model as its supervision.
# class DistillationLoss(nn.Module):
#     """
#         base_criterion: nn.Layer, the original criterion
#         teacher_model: nn.Layer, the teacher model as supervision
#         distillation_type: str, one of ['none', 'soft', 'hard']
#         alpha: float, ratio_1 of with_label_smooth loss (* (1-alpha)) and distillation loss( * alpha)
#         tao: float, temperature in distillation
#     """
#
#     def __init__(self, base_criterion, teacher_model, distillation_type='soft', alpha=0.25, tau=1):
#         super().__init__()
#         assert distillation_type in ['none', 'soft', 'hard']
#         self.base_criterion = base_criterion
#         self.teacher_model = teacher_model
#         self.type = distillation_type
#         self.alpha = alpha
#         self.tau = tau
#
#     def forward(self, inputs, outputs, targets):
#         """
#         Args:
#             inputs: tensor, the orginal model inputs
#             outputs: tensor, the outputs of the model
#             outputds_kd: tensor, the distillation outputs of the model,
#                          this is usually obtained by a separate branch
#                          in the last layer of the model
#             targets: tensor, the labels for the with_label_smooth criterion
#         """
#         global distillation_loss
#         outputs_kd = None
#
#         if not isinstance(outputs, torch.Tensor):
#             outputs, outputs_kd = outputs[0], outputs[1]
#
#         base_loss = self.base_criterion(outputs, targets)
#         with torch.no_grad():
#             teacher_outputs = self.teacher_model(inputs)
#
#         if self.type == 'none':
#             return base_loss
#         elif self.type == 'soft':
#             distillation_loss = F.kl_div(
#                 F.log_softmax(outputs_kd / self.tau, dim=1),
#                 F.log_softmax(teacher_outputs / self.tau, dim=1),
#                 reduction='sum') * (self.tau * self.tau) / outputs_kd.numel()
#         elif self.type == 'hard':
#             distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(axis=1))
#
#         loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
#
#         return loss

# class DistillationLoss:
#     def __init__(self, teacher_model):
#         self.student_loss = nn.CrossEntropyLoss()
#         self.distillation_loss = nn.KLDivLoss()
#         self.temperature = 1
#         self.alpha = 0.25
#         self.teacher_model = teacher_model
#
#     def __call__(self, inputs, student_logits, targets):
#         student_target_loss = self.student_loss(student_logits, targets)
#
#         with torch.no_grad():
#             teacher_logits = self.teacher_model(inputs)
#
#         distillation_loss = self.distillation_loss(F.log_softmax(student_logits / self.temperature, dim=1),
#                                                    F.softmax(teacher_logits / self.temperature, dim=1))
#
#         loss = (1 - self.alpha) * student_target_loss + self.alpha * distillation_loss
#
#         return loss


class Arc(nn.Module):
    # 初始化时我们需要传入特征数与类别数,传入2和7
    # 即认为在人脸检测时,只有正相关和不相关两种特征以及7个分类数
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn(feature_num, cls_num))

    # m为不同边之间的距离, 根据情况进行修改
    def forward(self, x, m=1, s=10):
        x_norm = F.normalize(x, dim=1)  # 在行上做标准化
        w_norm = F.normalize(self.w, dim=0)  # 在列上做标准化
        cosa = torch.matmul(x_norm, w_norm) / 10  # /10是为了防止梯度爆炸
        a = torch.arccos(cosa)  # 反求角度

        molecule = torch.exp(s * torch.cos(a + m))
        denominator = molecule + torch.sum(
            torch.exp(s * torch.cos(a)), dim=1, keepdim=True
        ) - torch.exp(s * torch.cos(a))  # 需要减去当前情况值

        arcsoftmax = torch.log(molecule / denominator)

        return arcsoftmax


class DCE(nn.Module):
    def __init__(self, num_class=7, reduction="mean", eps=1e-8):
        super(DCE, self).__init__()
        self.reduction = reduction
        self.num_class = num_class
        self.eps = eps

    def forward(self, prediction, target_label):
        y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class).float().to('cuda:0')
        y_pred = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(y_pred, self.eps, 1 - self.eps)

        pred_tmp = torch.sum(y_true * y_pred, axis=-1).reshape(-1, 1)

        avg = torch.mean(y_pred, dim=0)
        avg = avg.reshape(-1, 1)

        avg_ref = torch.matmul(y_true.type(torch.float), avg)

        pred = torch.where((pred_tmp >= avg_ref), pred_tmp, torch.zeros_like(pred_tmp))

        conf_idx = torch.where(pred != 0.)[0]

        if len(conf_idx) != 0:
            prun_targets = torch.argmax(torch.index_select(y_true, 0, conf_idx), dim=1)
            weighted_loss = F.cross_entropy(torch.index_select(prediction, 0, conf_idx), prun_targets,
                                            reduction=self.reduction)
        else:
            weighted_loss = F.cross_entropy(prediction, target_label)

        return weighted_loss


class AsymmetricLoss_MultiLabel(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss_MultiLabel, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    # x:input logits y:targets(multi-label binarized vector)
    def forward(self, x, y):
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


# focal loss的改进版,一种非对称的loss,即Asymmetric Loss
# 解决了多标签分类任务中,正负样本不平衡问题,标签错误问题
class AsymmetricLoss_SingleLabel(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(AsymmetricLoss_SingleLabel, self).__init__()
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []  # prevent gpu repeated memory allocation
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    # x:input logits y:targets(1-hot vector)
    def forward(self, inputs, target):
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs.clone())
        self.targets_classes = torch.zeros_like(inputs.clone()).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes.mul_(1 - self.eps).add_(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class WingLoss(nn.Module):
    def __init__(self, omega=0.01, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_2 = (y - y_hat).pow(2).sum(dim=-1, keepdim=False)
        # delta = delta_2.sqrt()
        delta = delta_2.clamp(min=1e-6).sqrt()
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - C
        )

        return loss.mean()


#########################################################################
class DistillKL(nn.Module):
    def __init__(self, temperature=1.0):
        super(DistillKL, self).__init__()
        self.T = temperature

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t.detach(), reduction='sum') * (self.T ** 2) / y_s.shape[0]

        return loss


class KL(nn.Module):
    def __init__(self, temperature, alpha, beta):
        super(KL, self).__init__()
        self.p = 2
        self.kd = DistillKL(temperature=temperature)
        self.alpha = alpha
        self.beta = beta

    def forward(self, o_s, o_t, g_s, g_t):
        loss = self.alpha * self.kd(o_s, o_t)
        loss += self.beta * sum([self.at_loss(f_s, f_t.detach()) for f_s, f_t in zip(g_s, g_t)])

        return loss

    def at_loss(self, f_s, f_t):
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class Orthogonal_Loss(nn.Module):
    def __init__(self, p=0.5):
        super(Orthogonal_Loss, self).__init__()
        self.p = p

    def forward(self, x_fc):
        x1, x2, x3 = x_fc
        l_12 = torch.matmul(x1, x2.T)
        l_13 = torch.matmul(x1, x3.T)
        l_23 = torch.matmul(x2, x3.T)
        loss1 = l_12 / (torch.norm(x1, 2) * torch.norm(x2, 2))
        loss2 = l_13 / (torch.norm(x1, 2) * torch.norm(x3, 2))
        loss3 = l_23 / (torch.norm(x2, 2) * torch.norm(x3, 2))
        loss = (loss1 + loss2 + loss3) / 3
        loss = loss * self.p

        return loss.mean()
