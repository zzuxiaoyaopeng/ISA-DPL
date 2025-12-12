import torch
import torch.nn as nn
from einops import rearrange


class LANet(nn.Module):
    def __init__(self, in_channel=512, ratio=32):
        super(LANet, self).__init__()
        self.con1x1_1 = nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1)
        self.relu = nn.ReLU()
        self.con1x1_2 = nn.Conv2d(in_channel // ratio, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.con1x1_1(x)
        x = self.relu(x)
        x = self.con1x1_2(x)
        x = self.sigmoid(x)

        return x

class Local_CNNs(nn.Module):
    def __init__(self, channel=512, ratio=32):
        super(Local_CNNs, self).__init__()
        # 构建本地分支
        self.LANet1 = LANet(channel, ratio)
        self.LANet2 = LANet(channel, ratio)

        # 最好建立MAD,当然Dropout也行
        self.drop1 = nn.Dropout(0.)
        self.drop2 = nn.Dropout(0.)

        # domain and pose invariance transform
        self.dp_conv1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.dp_conv2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.drop_dp = nn.Dropout(0.)

    def forward(self, x):
        # 构建残差
        short = x.clone()  # [bs,512,7,7]

        # 构建本地分支
        M1 = self.LANet1(x.clone())  # [bs,1,7,7]
        M2 = self.LANet2(x.clone())  # [bs,1,7,7]

        # drop处理(最好使用MAD)
        y1 = self.drop1(M1)  # [bs,1,7,7]
        y2 = self.drop2(M2)  # [bs,1,7,7]

        # max{y1,y2}
        Mout = torch.max(y1, y2)  # [bs,1,7,7]

        # channel-wise multiplication
        Xout = torch.matmul(Mout, short)  # [bs,512,7,7]

        # domain and pose invariance transform
        Xout = Xout + self.dp_conv1(Xout)  # [bs,512,7,7]
        Xout = Xout + self.dp_conv2(Xout)  # [bs,512,7,7]
        Xdp = self.drop_dp(Xout)  # [bs,512,7,7]

        # fuse short
        out = Xdp + short  # [bs,512,7,7]

        return out
