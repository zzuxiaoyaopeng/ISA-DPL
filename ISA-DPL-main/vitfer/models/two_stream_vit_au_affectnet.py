import torch
from torch import nn
from functools import partial


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor

    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention_img(nn.Module):
    def __init__(self, dim, in_chans, q_chanel, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.img_chanel = in_chans + 1
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x_img = x[:, :self.img_chanel, :]  # [bs,50,512]
        x_lm = x[:, self.img_chanel:, :]  # [bs,50,512]

        # bs,50,512
        B, N, C = x_img.shape
        # [bs,50,512]->[bs,50,2,8,64]->[2,bs,8,50,64]
        kv = self.kv(x_img).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [bs,8,50,64]
        k, v = kv.unbind(0)
        # [bs,50,8,64]->[bs,8,50,64]
        q = x_lm.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # [bs,8,50,50]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # [bs,8,50,64]->[bs,50,8,64]->[bs,50,512]
        x_img = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_img = self.proj(x_img)
        x_img = self.proj_drop(x_img)

        return x_img


class Attention_lm(nn.Module):
    def __init__(self, dim, in_chans, q_chanel, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.img_chanel = in_chans + 1
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x_img = x[:, :self.img_chanel, :]  # [bs,50,512]
        x_lm = x[:, self.img_chanel:, :]  # [bs,50,512]

        # bs,50,512
        B, N, C = x_lm.shape
        # [bs,50,512]->[bs,50,2,8,64]->[2,bs,8,50,64]
        kv = self.kv(x_lm).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [bs,8,50,64]
        k, v = kv.unbind(0)
        # [bs,50,8,64]->[bs,8,50,64]
        q = x_img.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # [bs,8,50,50]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # [bs,8,50,64]->[bs,50,8,64]->[bs,50,512]
        x_lm = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_lm = self.proj(x_lm)
        x_lm = self.proj_drop(x_lm)

        return x_lm


class Block(nn.Module):

    def __init__(self, dim, in_chans, q_chanel, num_heads, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.img_channel = in_chans + 1
        self.num_channels = in_chans + q_chanel + 2

        self.attn_img = Attention_img(dim, in_chans=in_chans, q_chanel=q_chanel,
                                      num_heads=num_heads, qkv_bias=qkv_bias,
                                      attn_drop=attn_drop, proj_drop=drop)
        self.attn_lm = Attention_lm(dim, in_chans=in_chans, q_chanel=q_chanel,
                                    num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        self.conv = nn.Conv1d(self.num_channels, self.num_channels, 1)

    def forward(self, x):
        bs, _, channel = x.shape
        x_img = x[:, :self.img_channel, :]  # [bs,50,512]
        x_lm = x[:, self.img_channel:, :]  # [bs,50,512]

        # 尝试对残差添加norm
        x_img = x_img + self.drop_path(self.attn_img(self.norm1(x)))
        x_img = x_img + self.drop_path(self.mlp1(self.norm2(x_img)))

        x_lm = x_lm + self.drop_path(self.attn_lm(self.norm3(x)))
        x_lm = x_lm + self.drop_path(self.mlp2(self.norm4(x_lm)))

        # [bs,100,512]
        x = torch.cat((x_img, x_lm), dim=1)

        # [bs,100,512]
        x = self.conv(x)

        return x


class PyramidBlock(nn.Module):

    def __init__(self, dim, in_chans, q_chanel, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.block_l = Block(
            dim=dim, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.block_m = Block(
            dim=dim // 2, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.block_s = Block(
            dim=dim // 4, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        n_channels = (in_chans + 1) + (q_chanel + 1)
        self.upsample_m = nn.ConvTranspose1d(n_channels, n_channels, kernel_size=2, stride=2)
        self.upsample_s = nn.ConvTranspose1d(n_channels, n_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x_l = x[0]
        x_m = x[1]
        x_s = x[2]

        x_l = self.block_l(x_l)  # [bs,100,512]
        x_m = self.block_m(x_m)  # [bs,100,256]
        x_s = self.block_s(x_s)  # [bs,100,128]

        x_m = self.upsample_s(x_s) + x_m
        x_l = x_l + self.upsample_m(x_m)
        x = [x_l, x_m, x_s]

        return x


class HyVisionTransformer(nn.Module):
    def __init__(self, in_chans=49, q_chanel=49, embed_dim=512, depth=8, num_heads=8, mlp_ratio=2.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, act_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        n_channels = (in_chans + 1) + (q_chanel + 1)
        self.downsample_m = nn.Conv1d(n_channels, n_channels, kernel_size=2, stride=2)
        self.downsample_s = nn.Conv1d(n_channels, n_channels, kernel_size=4, stride=4)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            self.blocks.append(PyramidBlock(dim=embed_dim, in_chans=in_chans, q_chanel=q_chanel,
                                            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                                            norm_layer=norm_layer, act_layer=act_layer))

        self.drop = nn.Dropout(0.)
        self.norm = norm_layer(embed_dim)

        # AU
        self.gap_AU11 = nn.AdaptiveAvgPool2d(1)
        self.gap_AU12 = nn.AdaptiveAvgPool2d(1)
        self.gap_AU13 = nn.AdaptiveAvgPool2d(1)
        self.gap_AU21 = nn.AdaptiveAvgPool2d(1)
        self.gap_AU22 = nn.AdaptiveAvgPool2d(1)
        self.gap_AU23 = nn.AdaptiveAvgPool2d(1)
        self.gap_AU3 = nn.AdaptiveAvgPool2d(1)
        self.norm_AU11 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.norm_AU12 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.norm_AU13 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.norm_AU21 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.norm_AU22 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.norm_AU23 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.norm_AU3 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.head_AU11 = nn.Linear(self.embed_dim, 4)
        self.head_AU12 = nn.Linear(self.embed_dim, 1)
        self.head_AU13 = nn.Linear(self.embed_dim, 4)
        self.head_AU21 = nn.Linear(self.embed_dim, 1)
        self.head_AU22 = nn.Linear(self.embed_dim, 1)
        self.head_AU23 = nn.Linear(self.embed_dim, 1)
        self.head_AU3 = nn.Linear(self.embed_dim, 14)

        self.deconv = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
        self.au_drop = nn.Dropout(0.)

    def AU_Branch(self, AU_x):
        AU_x = AU_x[:, 1:50, :]  # [bs,49,512]
        AU_x = AU_x.transpose(1, 2).view(-1, self.embed_dim, 7, 7)  # [bs,512,7,7]
        AU_x = self.deconv(AU_x)  # [bs,512,14,14]

        B, C, H, W = AU_x.shape
        AU11 = AU_x[:, :, :7, :7]  # 左眼
        AU12 = AU_x[:, :, :7, 4:10]  # 两眼中间部位
        AU13 = AU_x[:, :, :7, 7:]  # 右眼
        AU21 = AU_x[:, :, 5:12, :6]  # 左脸
        AU22 = AU_x[:, :, 4:10, 4:10]  # 鼻子
        AU23 = AU_x[:, :, 5:12, 8:]  # 右脸
        AU3 = AU_x[:, :, 6:, :]  # 嘴以及附近

        AU11 = self.head_AU11(self.norm_AU11(self.gap_AU11(AU11).squeeze(-1).squeeze(-1)))  # bsx4
        AU12 = self.head_AU12(self.norm_AU12(self.gap_AU12(AU12).squeeze(-1).squeeze(-1)))  # bsx1
        AU13 = self.head_AU13(self.norm_AU13(self.gap_AU13(AU13).squeeze(-1).squeeze(-1)))  # bsx4
        AU21 = self.head_AU21(self.norm_AU21(self.gap_AU21(AU21).squeeze(-1).squeeze(-1)))  # bsx1
        AU22 = self.head_AU22(self.norm_AU22(self.gap_AU22(AU22).squeeze(-1).squeeze(-1)))  # bsx1
        AU23 = self.head_AU23(self.norm_AU23(self.gap_AU23(AU23).squeeze(-1).squeeze(-1)))  # bsx1
        AU3 = self.head_AU3(self.norm_AU3(self.gap_AU3(AU3).squeeze(-1).squeeze(-1)))  # bsx14
        AU1257 = torch.maximum(AU11, AU13)  # 左右眼 bsx4
        AU6 = torch.maximum(AU21, AU23)  # 左右脸 bsx1

        AU_all = torch.cat(
            (AU1257[:, :2], AU12, AU1257[:, 2].view(B, -1), AU6, AU1257[:, 3].view(B, -1), AU22, AU3), dim=1)
        AU_all = self.au_drop(AU_all)


        return AU_all  # [bs,21]

    def forward(self, x, x_lm):
        # x/x_lm:[bs,50,512] new_x:[bs,100,512]
        new_x = torch.cat((x, x_lm), dim=1)

        # 特征金字塔操作
        new_x_l = new_x
        new_x_m = self.downsample_m(new_x)  # [bs,100,256]
        new_x_s = self.downsample_s(new_x)  # [bs,100,128]

        # 3种通道大小的特征图
        new_x_in = [new_x_l, new_x_m, new_x_s]

        # transformer blocks
        for i in range(self.depth):
            new_x_in = self.blocks[i](new_x_in)
            # 早期块中添加AU分支可能是有害的,这可能是由于早期块没有足够的语义信息用于AU估计,从而干扰了FER的底层特征学习
            # 随着AU分支位置的深入,性能逐渐提高,并在默认位置饱和,从最后两个块扩展可能会轻微地破坏FER的语义特征,导致性能下降
            # 8层第6个block输出用于AU Branch效果可能最好
            if i == 5:
                temp = new_x_in
                au_x = temp[0]  # [bs,100,512]
                au_x_out = self.AU_Branch(au_x)  # [bs,21]

        # 抽取channel为512的特征图,即[bs,100,512]
        new_x_l = new_x_in[0]

        # block后使用drop(有提升)
        new_x_l = self.drop(new_x_l)  # [bs,100,512]

        # block后进行LN
        new_x_l = self.norm(new_x_l)  # [bs,100,512]

        # img class token
        out = new_x_l[:, 0, :]  # [bs,512]
        img_embedding = new_x_l[:, 1:50, :]  # [bs,49,512]

        return out, au_x_out, img_embedding
