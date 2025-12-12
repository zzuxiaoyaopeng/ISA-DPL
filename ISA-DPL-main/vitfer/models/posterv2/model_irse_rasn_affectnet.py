import torch
import torch.nn as nn
from collections import namedtuple
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from vitfer.models.attn import SELayer


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                                             BatchNorm2d(depth))
        self.res_layer = Sequential(BatchNorm2d(in_channel),
                                    Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
                                    Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='irse'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['irse', 'ir_se'], "mode should be irse or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'irse':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512))
        # weight init
        self._initialize_weights()

        # RASN
        # self.bn1 = nn.BatchNorm2d(64)
        self.fc1_1 = nn.Linear(64 * 56 * 56, 64)
        self.fc1_2 = nn.Linear(64, 1)
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

        # self.bn2 = nn.BatchNorm2d(128)
        self.fc2_1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2_2 = nn.Linear(128, 1)
        self.relu2 = nn.ReLU()
        self.sigmoid2 = nn.Sigmoid()

        # self.bn3 = nn.BatchNorm2d(256)
        self.fc3_1 = nn.Linear(256 * 14 * 14, 256)
        self.fc3_2 = nn.Linear(256, 1)
        self.relu3 = nn.ReLU()
        self.sigmoid3 = nn.Sigmoid()

        # self.bn4 = nn.BatchNorm2d(512)
        self.fc4_1 = nn.Linear(512 * 7 * 7, 512)
        self.fc4_2 = nn.Linear(512, 1)
        self.relu4 = nn.ReLU()
        self.sigmoid4 = nn.Sigmoid()

        self.gn1 = nn.GroupNorm(4, 64)
        self.gn2 = nn.GroupNorm(8, 128)
        self.gn3 = nn.GroupNorm(16, 256)
        self.gn4 = nn.GroupNorm(32, 512)

        self.drop1 = nn.Dropout(0.05)
        self.drop2 = nn.Dropout(0.05)
        self.drop3 = nn.Dropout(0.05)
        self.drop4 = nn.Dropout(0.05)

        self.ratio = 0.1

    # 输入[bs,3,112,112]
    def forward(self, x):
        x = self.input_layer(x)  # [bs,64,112,112]
        p = self.ratio

        ##########################first stage###########################
        x = self.body[0:3](x)  # [bs,64,56,56]

        # affinity features
        fg = x  # [bs,64,56,56]
        fs = torch.mean(x, dim=0, keepdim=True)  # [1,64,56,56]
        short = fs  # [1,64,56,56]

        # attentive affinity features
        fi = self.gn1(fs)
        fi = torch.flatten(fi, 1)  # [1,64*56*56]
        fi = self.fc1_1(fi)  # [1,64]
        fi = self.relu1(fi)
        fi = self.fc1_2(fi)  # [1,1]
        weights = self.sigmoid1(fi)  # [1,1]
        fas = short * weights  # [1,64,56,56]
        fas = self.gn1(fas)
        fas = self.drop1(fas)

        # original features + affinity features
        x = fg + p * fas  # [bs,64,56,56]
        out1 = x  # [bs,64,56,56]

        ##########################second stage###########################
        x = self.body[3:7](x)  # [bs,128,28,28]

        # affinity features
        fg = x  # [bs,128,28,28]
        fs = torch.mean(x, dim=0, keepdim=True)  # [1,128,28,28]
        short = fs  # [1,128,28,28]

        # attentive affinity features
        fi = self.gn2(fs)
        fi = torch.flatten(fi, 1)  # [1,128*28*28]
        fi = self.fc2_1(fi)  # [1,128]
        fi = self.relu2(fi)
        fi = self.fc2_2(fi)  # [1,1]
        weights = self.sigmoid2(fi)  # [1,1]
        fas = short * weights  # [1,128,28,28]
        fas = self.gn2(fas)
        fas = self.drop2(fas)

        # original features + affinity features
        x = fg + p * fas  # [bs,128,28,28]
        out2 = x  # [bs,128,28,28]

        ##########################third stage###########################
        x = self.body[7:21](x)  # [bs,256,14,14]

        # affinity features
        fg = x  # [bs,256,14,14]
        fs = torch.mean(x, dim=0, keepdim=True)  # [1,256,14,14]
        short = fs  # [1,256,14,14]

        # attentive affinity features
        fi = self.gn3(fs)
        fi = torch.flatten(fi, 1)  # [1,256*14*14]
        fi = self.fc3_1(fi)  # [1,256]
        fi = self.relu3(fi)
        fi = self.fc3_2(fi)  # [1,1]
        weights = self.sigmoid3(fi)  # [1,1]
        fas = short * weights  # [1,256,14,14]
        fas = self.gn3(fas)
        fas = self.drop3(fas)

        # original features + affinity features
        x = fg + p * fas  # [bs,256,14,14]
        out3 = x  # [bs,256,14,14]

        ##########################fourth stage###########################
        x = self.body[21:](x)  # [bs,512,7,7]

        # affinity features
        fg = x  # [bs,512,7,7]
        fs = torch.mean(x, dim=0, keepdim=True)  # [1,512,7,7]
        short = fs  # [1,512,7,7]

        # attentive affinity features
        fi = self.gn4(fs)
        fi = torch.flatten(fi, 1)  # [1,512*7*7]
        fi = self.fc4_1(fi)  # [1,512]
        fi = self.relu4(fi)
        fi = self.fc4_2(fi)  # [1,1]
        weights = self.sigmoid4(fi)  # [1,1]
        fas = short * weights  # [1,512,7,7]
        fas = self.gn4(fas)
        fas = self.drop4(fas)

        # original features + affinity features
        x = fg + p * fas  # [bs,512,7,7]
        out4 = x  # [bs,512,7,7]

        return out1, out2, out3, out4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


def IR_50(input_size=[112, 112]):
    model = Backbone(input_size, 50, 'irse')
    return model


def IR_101(input_size=[112, 112]):
    model = Backbone(input_size, 100, 'irse')
    return model


def IR_152(input_size=[112, 112]):
    model = Backbone(input_size, 152, 'irse')
    return model


def IR_SE_50(input_size=[112, 112]):
    model = Backbone(input_size, 50, 'ir_se')
    return model


def IR_SE_101(input_size=[112, 112]):
    model = Backbone(input_size, 100, 'ir_se')
    return model


def IR_SE_152(input_size=[112, 112]):
    model = Backbone(input_size, 152, 'ir_se')
    return model


if __name__ == '__main__':
    inputs = torch.randn(32, 3, 112, 112)
    model = IR_50()
    out = model(inputs)
    print(model)
