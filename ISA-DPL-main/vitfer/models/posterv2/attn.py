import torch
import torch.nn as nn
import torch.nn.functional as F


# SE注意力模块，针对[bs,c,h,w]的数据
class SELayer(nn.Module):
    def __init__(self, channel=512, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


# 同SELayer,只不过针对输入为[bs,c]的数据
class SE_block(nn.Module):
    def __init__(self, channel=512, r=1):
        super().__init__()
        self.fc1 = nn.Linear(channel, channel//r)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel//r, channel)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.sigmod(x1)
        x = x * x1

        return x


# ECA注意力模块Efficient Channel Attention
# 其与SE模块唯一的区别在于：没有将通道注意力向量压缩后再放大的全连接层,而是直接将其与特征图进行加权运算
class eca_layer(nn.Module):
    def __init__(self, kernel=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel, padding=(kernel - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:[bs,c,h,w]
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, channel=512, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool = self.max_pool(x).view([b, c])
        avg_pool = self.avg_pool(x).view([b, c])

        max_fc = self.fc(max_pool)
        avg_fc = self.fc(avg_pool)

        out = max_fc + avg_fc
        out = self.sigmoid(out).view([b, c, 1, 1])

        return out * x


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 压缩通道提取空间信息
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 经过卷积提取空间注意力权重
        pool_out = torch.cat([max_out, avg_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)

        return out * x


# CBAM通道与空间混合的注意力机制
class CBAM(nn.Module):
    def __init__(self, channel=512):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


# GCA:输入为[bs,512,7,7]
class GCA(nn.Module):
    def __init__(self, channel=512, kernel=7):
        super(GCA, self).__init__()
        # 卷积代替全局平均池化
        self.global_conv = nn.Conv2d(channel, channel, kernel_size=kernel)
        self.conv1 = nn.Conv2d(channel, channel//2, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel//2, channel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        short = x  # [bs,512,7,7]
        x = self.global_conv(x)  # [bs,512,1,1]
        x = self.conv1(x)  # [bs,256,1,1]
        x = self.relu(x)  # [bs,256,1,1]
        x = self.conv2(x)  # [bs,512,1,1]
        x = self.sigmoid(x)  # [bs,512,1,1]

        short2 = x  # [bs,512,1,1]
        x = torch.mean(x, dim=0, keepdim=True)  # [1,512,1,1]

        Sgca = short2 * x  # [bs,512,1,1]
        eps = 1e-7  # 防止反向传播出错
        Sgca = torch.sqrt(Sgca + eps)
        Sgca = Sgca * 2

        # fuse short
        out = short * Sgca  # [bs,512,7,7]

        return out


# GCA_block:输入为[bs,512]
# 针对时序图像,就是同一类表情图片组成的bs
class GCA_block(nn.Module):
    def __init__(self):
        super(GCA_block, self).__init__()
        # 512-64-512最佳
        self.fc1 = nn.Linear(512, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # short:[bs,512]
        short = x
        # [bs,512]->[bs,256]
        x = self.fc1(x)
        x = self.relu(x)
        # [bs,256]->[bs,512]
        x = self.fc2(x)
        x = self.sigmoid(x)

        # short2:[bs,512]
        short2 = x
        # [bs,512]->[pure_embedding,512]
        x = torch.mean(x, dim=0).unsqueeze(dim=0)

        # Sgca:[bs,512]
        Sgca = short2 * x
        eps = 1e-7  # 防止反向传播出错
        Sgca = torch.sqrt(Sgca + eps)
        Sgca = Sgca * 2

        # fuse short
        out = short * Sgca

        return out


class no_local(nn.Module):
    def __init__(self, channel=512, length=7):
        super(no_local, self).__init__()
        self.channel = channel
        self.hw = length
        self.conv1 = nn.Conv2d(self.channel, self.channel // 2, 1)
        self.conv2 = nn.Conv2d(self.channel, self.channel // 2, 1)
        self.conv3 = nn.Conv2d(self.channel, self.channel // 2, 1)
        self.conv4 = nn.Conv2d(self.channel // 2, self.channel, 1)

    # x:[bs,512,7,7]
    def forward(self, x):
        short = x  # [bs,512,7,7]
        x1 = self.conv1(x.clone())  # [bs,256,7,7]
        x2 = self.conv2(x.clone())  # [bs,256,7,7]
        x3 = self.conv3(x.clone())  # [bs,256,7,7]
        x1 = x1.reshape(-1, self.channel // 2, self.hw * self.hw).permute(0, 2, 1)  # [bs,49,256]
        x2 = x2.reshape(-1, self.channel // 2, self.hw * self.hw)  # [bs,256,49]
        x3 = x3.reshape(-1, self.channel // 2, self.hw * self.hw).permute(0, 2, 1)  # [bs,49,256]
        x4 = x1 @ x2  # [bs,49,49]
        # x4 = F.softmax(x4, dim=-1)
        x4 = x4.softmax(dim=-1)
        x5 = x4 @ x3  # [bs,49,256]
        x5 = x5.permute(0, 2, 1)  # [bs,256,49]
        x5 = x5.reshape(-1, self.channel // 2, self.hw, self.hw)  # [bs,256,7,7]
        x5 = self.conv4(x5)  # [bs,512,7,7]
        x = torch.add(short, x5)

        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp=512, oup=512, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# AttentionRefinementModule模块(通道注意力)
# ARM使用在上下文路径中,用于优化每一阶段的特征,使用全局平均池化指导特征学习
class ARM(torch.nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.bn(x)
        x = self.sigmoid(x)
        x = torch.mul(input, x)

        return x


# FeatureFusionModule模块
# FFM用于融合CP和SP提供的输出特征,由于两路特征并不相同,所以不能对这两部分特征进行简单的加权
# SP提供的特征是低层次的8×down,CP提供的特征是高层语义的32×down
class FFM(torch.nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        self.in_channels = in_channels
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, classes, kernel_size=1),
            # nn.BatchNorm2d(classes),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1 = nn.Conv2d(classes, classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(classes, classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))

        feature = self.convblock(x)
        x = self.avgpool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)

        return x


# Selective Kernel Attention(SKA)
# SE的升级版
class SKA(nn.Module):
    """
    Args:
        features: input channel dimensionality.
        WH: input spatial dimensionality, used for GAP kernel size.
        M: the number of branchs.
        G: num of convolution groups.
        r: the radio for compute d, the length of z.
        stride: stride, default pure_embedding.
        L: the minimum dim of the vector z in paper, default 32.
    """

    def __init__(self, features, WH=0, M=3, G=1, r=1, stride=1, L=32):
        super(SKA, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))
        # self.gap = nn.AvgPool2d(int(WH/stride))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            fea = fea.unsqueeze(dim=1)
            # fea = conv(x).unsqueeze_(dim=pure_embedding)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)

        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)

        return fea_v


# 详细见RANet
class SRAM(torch.nn.Module):
    def __init__(self, channel=512, r=2):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel // r, 1)

        # 1x3,3x3,3x1 separable convolutions
        c = channel // r
        self.conv_1 = nn.Conv2d(c, c, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_2 = nn.Conv2d(c, c, kernel_size=(3, 3), stride=1, padding=1)
        self.conv_3 = nn.Conv2d(c, c, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        # pw
        self.conv_pw1 = nn.Conv2d(c, c // r, 1)
        self.conv_pw2 = nn.Conv2d(c, c // r, 1)
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

    def forward(self, x):
        short = x  # [bs,512,7,7]

        x = self.conv(x)
        x1 = self.conv_1(x)  # [bs,256,7,7]
        x1 = self.relu1(x1)
        x2 = self.conv_2(x)  # [bs,256,7,7]
        x2 = self.relu2(x2)
        x3 = self.conv_3(x)  # [bs,256,7,7]
        x3 = self.relu3(x3)
        x = x1 + x2 + x3  # [bs,256,7,7]

        G = self.conv_pw1(x)  # [bs,128,7,7]
        G = self.relu4(G)
        G = G.reshape(-1, 128, 49)
        G = G.transpose(-1, -2)  # [bs,49,128]

        Q = self.conv_pw2(x)  # [bs,128,7,7]
        Q = self.relu5(Q)
        Q = Q.reshape(-1, 128, 49)  # [bs,128,49]

        X = G @ Q  # [bs,49,49]
        X = F.softmax(X, dim=-1)  # [bs,49,49]

        short = short.reshape(-1, 512, 49)
        out = short @ X  # [bs,512,49]
        out = out.reshape(-1, 512, 7, 7)

        return out


class CRAM(torch.nn.Module):
    def __init__(self, channel=256):
        super().__init__()
        self.c = channel

    def forward(self, x):
        short = x  # [bs,256,14,14]

        x = x.reshape(-1, self.c, 196)  # [bs,256,196]
        U = x[:, :, 0:98]  # [bs,256,98]
        V = x[:, :, 98:]  # [bs,256,98]

        V = V.transpose(-1, -2)  # [bs,98,256]
        C = U @ V  # [bs,256,256]
        Rc = C.unsqueeze(dim=2)  # [bs,256,1,256]
        Rv = C.unsqueeze(dim=2)  # [bs,256,1,256]
        R = torch.cat([Rc, Rv], dim=1)  # [bs,512,1,256]

        R = torch.mean(R, dim=1, keepdim=True)  # [bs,1,1,256]
        R = R.reshape(-1, self.c, 1, 1)  # [bs,256,1,1]
        out = R * short  # [bs,256,14,14]

        return out


class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)



if __name__ == '__main__':
    input = torch.rand(32, 512, 7, 7)
    model = SRAM()
    out = model(input)
    print(out.shape)
