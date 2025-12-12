import os
import sys
import torch.nn as nn
import numpy as np
import torch.utils.data
import random
from torch.backends import cudnn
from torchvision import transforms
from tqdm import tqdm  # 进度条管理
from datasets.affectnet_dataset_complex import AffectNet
from datasets.raf_au_dataset import Dataset_RAF_AU
from vitfer.utils import accuracy, save_best_checkpoint, save_last_checkpoint
from vitfer.logger import Logger
from vitfer.misc import AverageMeter
from vitfer.models.model_poster_vit_au_arbex_posterv2_affect import Poster
from vitfer.sam import SAM
from vitfer.losses import LabelSmoothingCrossEntropy, CenterLoss, AnchorLoss
from vitfer.adan_optimizer import Adan
from vitfer.models.arbex import Anchors
from vitfer.sampler import ImbalancedDatasetSampler
from sklearn.metrics import f1_score, precision_recall_fscore_support
from vitfer.plot_confusion_matrix import ConfusionMatrix


def normalized_entropy(x):
    norm = torch.log(torch.tensor([len(x[0])])).item()
    h = -torch.sum(torch.log(x) * x, -1) / norm
    return h


if __name__ == '__main__':
    # 模型含有上采样可能会导致结果无法复现?
    # seed = 123
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True

    # 3407或者42的倍数
    torch.manual_seed(123)
    cudnn.benchmark = True

    # Logger
    logger = Logger(os.path.join("trained_weights", 'log.txt'), title='AffectNet')
    logger.set_names(['Epoch', 'Train Loss', 'Test Loss', 'Test Acc', 'Max Acc', 'Precision', 'Recall', 'F1'])

    # variable settings
    lr = 0.000002  # 0.000002
    classes = 7
    workers = 2
    start_epoch = 1
    best_acc = 0.0
    total_epochs = 300
    flood_level = 0.
    resume = ''
    trust = 0.015
    decay = 1e-2
    labels_info_7 = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    labels_info_8 = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

    # mean & std settings
    # mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    # data_transforms
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(scale=(0.02, 0.1)),
    ])
    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    au_transform_train = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # AffectNet验证共4000个数据,其中7分类共3500个,加上第八分类共4000个
    # (total:287647  0:74873 1:134412 2:25459 3:14090 4:6378 5:3803 6:24882 7:3750)
    # root = '../datasets/AffectNet/data/Manually_Annotated/Manually_Annotated_Images'
    # file_list_train = '../datasets/AffectNet/labels/training.csv'
    # file_list_val = '../datasets/AffectNet/labels/validation.csv'

    # root = '../datasets/AffectNet/data_cleaned/data'
    # file_list_train = '../datasets/AffectNet/data_cleaned/labels/training.csv'
    # file_list_val = '../datasets/AffectNet/data_cleaned/labels/validation.csv'

    root = 'D:\Pycharm WorkingSpace\cl\Ada-CM\datasets\AffectNet\data\Manually_Annotated\Manually_Annotated_Images'
    file_list_train = 'D:\Pycharm WorkingSpace\cl\Ada-CM\datasets\AffectNet\labels\\training.csv'
    file_list_val = 'D:\Pycharm WorkingSpace\cl\Ada-CM\datasets\AffectNet\labels\\validation.csv'
    data_train = AffectNet(root, file_list_train, num_classes=classes, transform=transform_train, phase='train')
    data_val = AffectNet(root, file_list_val, num_classes=classes, transform=transform_val, phase='val')
    print(len(data_train))
    print(len(data_val))
    sampler = ImbalancedDatasetSampler(data_train)
    data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=43, sampler=sampler, num_workers=workers)
    data_loader_val = torch.utils.data.DataLoader(data_val, batch_size=64, shuffle=False, num_workers=workers)


    # AU datasets(3801)
    au_root = '../datasets/raf_au/aligned/'
    au_labels_path = '../datasets/raf_au/labels.txt'
    data_au_train = Dataset_RAF_AU(au_root, au_labels_path, transform=au_transform_train)
    print(len(data_au_train))
    data_loader_au_train = torch.utils.data.DataLoader(data_au_train, batch_size=9, shuffle=True, num_workers=workers)

    # build model
    model = Poster(num_class=classes)
    print('Train set size:', data_train.__len__())
     # set GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # optimizer
    base_optimizer = Adan
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, adaptive=False, lr=lr, weight_decay=decay)

    # scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # loss func
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)
    lsce_criterion.to(device)

    # AU 1 2 4 5 6 7 9 10 12 14 15 16 17 18 20 22 23 24 25 26 27 共21种
    pos_weight_21 = np.array([3.04216867, 4.40402685, 1.42384106, 3.5033557, 8.63157895,
                              11.74050633, 4.62290503, 2.0407855, 2.38603869, 40.9375,
                              14.66536965, 4.98216939, 7.37006237, 39.66666667, 19.02985075,
                              11.94533762, 38.86138614, 20.18947368, 0.52442257, 2.96259843, 4.27653997])
    pos_weight = torch.from_numpy(pos_weight_21)
    criterion_au = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_au.to(device)

    # arbex
    confidence_fn = lambda x: 1 - normalized_entropy(x)
    loss_fn_mu = AnchorLoss(dim_emb=512)
    loss_fn_mu.to(device)
    loss_fn_center = CenterLoss(dim_emb=512)
    loss_fn_center.to(device)
    anchors = Anchors(size_emb=512, n_classes=classes, n_anchors=1)
    anchors.to(device)

    # 如果指定了上次训练保存的权重文件地址,则接着上次结果接着训练
    if resume != "":
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        anchors.load_state_dict(checkpoint['anchors'])
        best_acc = checkpoint['best_acc']
        epoch_current = checkpoint['epoch']
        restart_epoch = epoch_current + 1
        print("the training process from epoch{}...".format(restart_epoch))

    # 计算模型参数量
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # 开始训练与测试
    for i in range(start_epoch, total_epochs):
        train_loss = AverageMeter()
        val_loss = AverageMeter()
        top1 = AverageMeter()

        # 训练
        model.train()
        anchors.train()
        train_bar = tqdm(data_loader_train, file=sys.stdout)  # 会自动输出每轮训练进度
        iter_AU_loader_train = iter(data_loader_au_train)
        for step, data in enumerate(train_bar):
            optimizer.zero_grad()

            # load AU data
            try:
                AU_imgs, AU_targets = next(iter_AU_loader_train)
            except:
                iter_AU_loader_train = iter(data_loader_au_train)
                AU_imgs, AU_targets = next(iter_AU_loader_train)  # 初始AU_targets为列表

            # AU数据集以及标签
            AU_imgs = AU_imgs.to(device)
            AU_targets = np.array(AU_targets, dtype='int32').T
            AU_targets = torch.tensor(AU_targets)
            AU_targets = AU_targets.to(device)

            # RAF-DB数据集与标签
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            bs = images.shape[0]

            # concat FER and AU imgs
            total_imgs = torch.cat((images, AU_imgs), dim=0)

            # first forward-backward pass
            outputs, embeddings, au_outputs = model(total_imgs)

            ce_loss = criterion(outputs[0:bs], targets.long())
            lsce_loss = lsce_criterion(outputs[0:bs], targets.long())
            bce_loss = criterion_au(au_outputs[bs:], AU_targets.float())

            # arbex
            prob_dist = torch.softmax(outputs[0:bs] / 1.0, -1)  # label prob distribution
            confidence = confidence_fn(prob_dist).view(-1, 1)
            distances = anchors(embeddings[0:bs])  # [batch, n_classes, n_anchors]
            mu_loss = loss_fn_mu(anchors.get_anchors())  # keep anchors apart
            center_loss = loss_fn_center(distances, targets, confidence)  # keep embeddings in the right cluster
            arbex_loss = mu_loss * trust + center_loss * trust

            loss = 2 * lsce_loss + ce_loss + bce_loss + arbex_loss
            loss = (loss - flood_level).abs() + flood_level
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            outputs, embeddings, au_outputs = model(total_imgs)

            ce_loss = criterion(outputs[0:bs], targets.long())
            lsce_loss = lsce_criterion(outputs[0:bs], targets.long())
            bce_loss = criterion_au(au_outputs[bs:], AU_targets.float())

            # arbex
            prob_dist = torch.softmax(outputs[0:bs] / 1.0, -1)  # label prob distribution
            confidence = confidence_fn(prob_dist).view(-1, 1)
            distances = anchors(embeddings[0:bs])  # [batch, n_classes, n_anchors]
            mu_loss = loss_fn_mu(anchors.get_anchors())  # keep anchors apart
            center_loss = loss_fn_center(distances, targets, confidence)  # keep embeddings in the right cluster
            arbex_loss = mu_loss * trust + center_loss * trust

            loss = 2 * lsce_loss + ce_loss + bce_loss + arbex_loss
            loss = (loss - flood_level).abs() + flood_level
            loss.backward()
            optimizer.second_step(zero_grad=True)

            # train loss
            train_loss.update(loss.item(), images.size(0))

        # 更新学习率
        scheduler.step()

        # 验证测试
        pre_labels = []
        gt_labels = []
        if classes == 7:
            confusion_matrix = ConfusionMatrix(classes, labels_info_7)  # confusion matrix
        else:
            confusion_matrix = ConfusionMatrix(classes, labels_info_8)  # confusion matrix
        model.eval()
        with torch.no_grad():
            val_bar = tqdm(data_loader_val, file=sys.stdout)
            for step, data in enumerate(val_bar):
                val_images, val_targets = data
                val_images = val_images.to(device)
                val_targets = val_targets.to(device)

                val_outputs, _, _ = model(val_images)
                loss_2 = criterion(val_outputs, val_targets.long()).mean()
                val_loss.update(loss_2.item(), val_images.size(0))

                prec1, prec5 = accuracy(val_outputs, val_targets, topk=(1, 5))
                top1.update(prec1.item(), val_images.size(0))

                _, predicts = torch.max(val_outputs, 1)
                pre_labels += predicts.cpu().tolist()
                gt_labels += val_targets.cpu().tolist()

                confusion_matrix.update(predicts.cpu().numpy(), val_targets.cpu().numpy())

        # 输入真实标签和预测标签,得到精确率,召回率,f1值
        precision, recall, f1, _ = precision_recall_fscore_support(gt_labels, pre_labels, average='macro')

        # 输出当前epoch中训练集,测试集的平均损失,以及当前epoch的准确度
        print("[Epoch{}] average_train_loss:{:.3f}, average_val_loss:{:.3f}, accuracy:{:.3f}, "
              "precision:{:.3f}, recall:{:.3f}, f1:{:.3f}"
              .format(i, train_loss.avg, val_loss.avg, top1.avg, precision, recall, f1))

        # 保存权重
        is_best = top1.avg > best_acc
        best_acc = max(top1.avg, best_acc)
        state = {
            'epoch': i,
            'state_dict': model.state_dict(),
            'acc': top1.avg,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'lr': optimizer.state_dict()['param_groups'][0]['lr'],
            'scheduler': scheduler.state_dict(),
            'anchors': anchors.state_dict()
        }
        # 保存最佳权重(不断覆盖)
        save_best_checkpoint(state, is_best)

        # 保存最后一轮权重(不断覆盖)
        save_last_checkpoint(state)

        # 保存日志信息
        logger.append([i, train_loss.avg, val_loss.avg, top1.avg, best_acc, precision, recall, f1])

        # 混淆矩阵(每一轮)
        confusion_matrix.plot(i)

    # 结束处理
    logger.close()
