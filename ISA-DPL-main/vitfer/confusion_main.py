import torch
import sys
import argparse
from tqdm import tqdm
from torchvision import transforms
import torchvision

# 导入工具类
from vitfer.plot_confusion_matrix import ConfusionMatrix

# 导入数据集类
from datasets.affectnet_dataset_complex import AffectNet
from datasets.raf_dataset_poster import Dataset_RAF

# 导入模型 (使用别名防止冲突)
from vitfer.models.model_poster_vit_au_arbex_posterv2_affect import Poster as PosterAffect
from vitfer.models.model_poster_vit_au_arbex_posterv2_bk4 import Poster as PosterRAF
from vitfer.models.model_poster_vit_au_arbex_posterv2_sfew import Poster as PosterSFEW


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(f"Start evaluation on {args.dataset} using device: {device}")

    # ==========================
    # 1. 配置: AffectNet
    # ==========================
    if args.dataset == 'affectnet':
        # 路径配置 (请修改这里)
        root_path = r'D:\Pycharm WorkingSpace\cl\Ada-CM\datasets\AffectNet\data\Manually_Annotated\Manually_Annotated_Images'
        list_path = r'D:\Pycharm WorkingSpace\cl\Ada-CM\datasets\AffectNet\labels\validation.csv'
        weights_path = './checkpoint/model_best_affectnet_8cls.pth'

        classes = 8
        labels_info = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
        if classes == 8:
            labels_info.append('Contempt')

        # Transform
        transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Dataset & Loader
        data_val = AffectNet(root_path, list_path, num_classes=classes, transform=transform_val, phase='val')

        # Model
        model = PosterAffect(num_class=classes)


    elif args.dataset == 'rafdb':
        # 路径配置 (请修改这里)
        img_path = '../datasets/test_images/'
        label_path = '../datasets/test_labels.txt'
        weights_path = './checkpoint/model_best_rafdb.pth'

        classes = 7
        labels_info = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

        # Transform
        transform_val = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Dataset & Loader
        data_val = Dataset_RAF(img_path, label_path, transform=transform_val, phase='val')

        # Model
        model = PosterRAF(num_class=classes)

    # ==========================
    # 3. 配置: SFEW
    # ==========================
    elif args.dataset == 'sfew':
        # 路径配置 (请修改这里)
        data_dir = '../datasets/SFEW2.0/val'
        weights_path = './checkpoint/model_best_sfew.pth'

        classes = 7
        labels_info = ['Anger', 'Disgust', 'Fear', 'Neutral', 'Happiness', 'Sadness', 'Surprise']

        # Transform (注意：SFEW 使用 224x224)
        mean_au = (0.5, 0.5, 0.5)
        std_au = (0.5, 0.5, 0.5)
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_au, std=std_au),
        ])

        # Dataset & Loader
        data_val = torchvision.datasets.ImageFolder(data_dir, transform=transform_val)

        # Model
        model = PosterSFEW(num_class=classes)

    else:
        raise ValueError("Invalid dataset name. Choose from: affectnet, rafdb, sfew")

    # ==========================
    # 通用执行逻辑
    # ==========================
    data_loader_val = torch.utils.data.DataLoader(data_val, batch_size=64, shuffle=False)

    # 初始化混淆矩阵工具
    confusion_matrix = ConfusionMatrix(classes, labels_info)


    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'], strict=False)


    model.to(device)
    model.eval()

    pre_labels = []
    gt_labels = []


    with torch.no_grad():
        val_bar = tqdm(data_loader_val, file=sys.stdout)
        for step, data in enumerate(val_bar):
            # 处理不同 Dataset 返回数据格式可能微小的差异 (通常是 images, targets)
            val_images, val_targets = data
            val_images = val_images.to(device)
            val_targets = val_targets.to(device)

            # 前向传播
            val_outputs, _, _ = model(val_images)
            _, predicts = torch.max(val_outputs, 1)

            # 记录结果
            pre_labels += predicts.cpu().tolist()
            gt_labels += val_targets.cpu().tolist()
            confusion_matrix.update(predicts.cpu().numpy(), val_targets.cpu().numpy())

    # # 绘制混淆矩阵
    # plot_title = args.dataset if args.dataset != 'affectnet' else 000  # 保持原代码风格
    confusion_matrix.plot(args.dataset)
    print(f"Done for {args.dataset}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Poster Model on different datasets')
    parser.add_argument('--dataset', type=str, default='sfew',
                        choices=['affectnet', 'rafdb', 'sfew'],
                        help='Choose the dataset to evaluate: affectnet, rafdb, or sfew')

    args = parser.parse_args()
    main(args)