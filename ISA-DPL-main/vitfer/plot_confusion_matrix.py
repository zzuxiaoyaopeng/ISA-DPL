import itertools
import matplotlib.pyplot as plt  # 绘图库
import numpy as np
import os
from prettytable import PrettyTable


def plot_confusion_matrix(cm, labels_name, title, acc):
    cm = cm / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_class = np.array(range(len(labels_name)))  # 获取标签的间隔数
    plt.xticks(num_class, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_class, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('Target')
    plt.xlabel('Prediction')
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.tight_layout()
    plt.savefig(os.path.join('./Confusion_matrix/rafdb', "acc" + str(acc) + ".png"), format='png')
    plt.show()


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率=混淆矩阵对角线的元素之和 / 总样本数

        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        kappa = round((po - pe) / (1 - pe), 3)

        # precision, recall, specificity
        table = PrettyTable()  # 创建一个表格
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):  # 精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.  # 每一类准确度
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity])

        # print(table)
        return acc

    def plot(self, epoch):  # 绘制混淆矩阵
        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        acc_total = "{:.2f}".format(self.summary() * 100)
        plt.title('Confusion matrix (acc=' + acc_total + ')')

        # 在图中标注数量/概率信息
        # 每个类别的准确率公式为TP / (TP + FP)
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])   # 当前位置的真实预测数量, TP
                n1 = np.sum(matrix[:, x])  # 列和,预测为类别x的所有样本数量, TP + FP
                n2 = np.sum(matrix[y, :])  # 行和,实际为类别y所有样本数量, TP + FN
                acc_per = "{:.2f}".format(info / n1 * 100)
                # recall = "{:.2f}".format(info / n2 * 100)
                # print(info, n1)
                plt.text(x, y, acc_per,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")

        plt.tight_layout()  # 布局更为紧凑
        # plt.show()
        matrix_name = './confusion_matrix/tmp/{0}.png'.format(epoch)
        plt.savefig(matrix_name)
        plt.clf()  # 清空画布以防重复叠加
