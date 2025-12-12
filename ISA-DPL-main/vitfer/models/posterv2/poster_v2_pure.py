import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange


# [B,C,H,W]->[B,H,W,C]
def _to_channel_last(x):
    return x.permute(0, 2, 3, 1)

# [B,C,H,W]->[B,W,C,H]
def _to_channel_first(x):
    return x.permute(0, 3, 1, 2)

# [B,C,H,W]->[B,pure_embedding,H*W,num_heads,dim_head]
def _to_query(x, N, num_heads, dim_head):
    B = x.shape[0]
    x = x.reshape(B, 1, N, num_heads, dim_head).permute(0, 1, 3, 2, 4)
    return x

def window_partition(x, window_size, h_w, w_w):
    B, H, W, C = x.shape
    x = x.view(B, h_w, window_size, w_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows

def window_reverse(windows, window_size, H, W, h_w, w_w):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, h_w, w_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

# 尝试将此处的norm放到ffn中 不行
# 尝试LN后加drop 不行
class window(nn.Module):
    def __init__(self, window_size, dim):
        super(window, self).__init__()
        self.window_size = window_size
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = self.norm(x)
        shortcut = x
        h_w = int(torch.div(H, self.window_size).item())
        w_w = int(torch.div(W, self.window_size).item())
        x_windows = window_partition(x, self.window_size, h_w, w_w)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        return x_windows, shortcut

class WindowAttentionGlobal(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., ):
        super().__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads)
        self.scale = qk_scale or head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, q_global):
        B_, N, C = x.shape
        B = q_global.shape[0]

        head_dim = int(torch.div(C, self.num_heads).item())
        B_dim = int(torch.div(B_, B).item())

        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q_global = q_global.repeat(1, B_dim, 1, 1, 1)
        q = q_global.reshape(B_, self.num_heads, N, head_dim)
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# layer_scale值可以尝试变化,默认1e-5 不行
# 尝试不带有layer_scale 不行
# mlp_ratio尝试2或8 不行
# 改drop值 不行
# 尝试两个norm 不行
# norm 改为1e-6 不行
class feedforward(nn.Module):
    def __init__(self, dim, window_size, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0., layer_scale=None):
        super(feedforward, self).__init__()
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
        self.window_size = window_size
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, attn_windows, shortcut):
        B, H, W, C = shortcut.shape
        h_w = int(torch.div(H, self.window_size).item())
        w_w = int(torch.div(W, self.window_size).item())
        x = window_reverse(attn_windows, self.window_size, H, W, h_w, w_w)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm(x)))

        return x


class PosterV2(nn.Module):
    def __init__(self, window_size=[28, 14, 7], num_heads=[2, 4, 8], dims=[64, 128, 256]):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = []
        for num_head, dim in zip(num_heads, dims):
            self.dim_head.append(int(torch.div(dim, num_head).item()))
        self.window_size = window_size
        self.N = [win * win for win in window_size]

        self.conv1 = nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=dims[1], out_channels=dims[1], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=dims[2], out_channels=dims[2], kernel_size=3, stride=2, padding=1)
        self.last_face_conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.attn1 = WindowAttentionGlobal(dim=dims[0], num_heads=num_heads[0], window_size=window_size[0])
        self.attn2 = WindowAttentionGlobal(dim=dims[1], num_heads=num_heads[1], window_size=window_size[1])
        self.attn3 = WindowAttentionGlobal(dim=dims[2], num_heads=num_heads[2], window_size=window_size[2])

        self.window1 = window(window_size=window_size[0], dim=dims[0])
        self.window2 = window(window_size=window_size[1], dim=dims[1])
        self.window3 = window(window_size=window_size[2], dim=dims[2])

        dpr = [x.item() for x in torch.linspace(0, 0.5, 5)]  # (0,0.5,5):[0.0, 0.125, 0.25, 0.375, 0.5]
        self.ffn1 = feedforward(dim=dims[0], window_size=window_size[0], layer_scale=1e-5, drop_path=dpr[0])
        self.ffn2 = feedforward(dim=dims[1], window_size=window_size[1], layer_scale=1e-5, drop_path=dpr[1])
        self.ffn3 = feedforward(dim=dims[2], window_size=window_size[2], layer_scale=1e-5, drop_path=dpr[2])

        # self.embed_q = nn.Sequential(nn.Conv2d(dims[0], 512, kernel_size=3, stride=2, padding=1),
        #                              nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1))
        # self.embed_k = nn.Sequential(nn.Conv2d(dims[1], 512, kernel_size=3, stride=2, padding=1))
        # self.embed_v = nn.Sequential(nn.Conv2d(dims[2], 512, kernel_size=1),
        #                              nn.Identity())

    def forward(self, x, x_face):
        # [bs,64,56,56],[bs,128,28,28],[bs,256,14,14]
        x_ir1, x_ir2, x_ir3 = x
        # [bs,64,28,28],[bs,128,14,14],[bs,512,7,7]
        x_face1, x_face2, x_face3 = x_face

        x_face3 = self.last_face_conv(x_face3)  # [bs,256,7,7]
        x_face1, x_face2, x_face3 = _to_channel_last(x_face1), _to_channel_last(x_face2), _to_channel_last(x_face3)

        q1 = _to_query(x_face1, self.N[0], self.num_heads[0], self.dim_head[0])  # [bs,1,2,784,32]
        q2 = _to_query(x_face2, self.N[1], self.num_heads[1], self.dim_head[1])  # [bs,1,4,196,32]
        q3 = _to_query(x_face3, self.N[2], self.num_heads[2], self.dim_head[2])  # [bs,1,8,49,32]

        # [bs,64,28,28],[bs,128,14,14],[bs,256,7,7]
        x_ir1, x_ir2, x_ir3 = self.conv1(x_ir1), self.conv2(x_ir2), self.conv3(x_ir3)

        x_window1, shortcut1 = self.window1(x_ir1)  # [bs,784,64],[bs,28,28,64]
        x_window2, shortcut2 = self.window2(x_ir2)  # [bs,196,128],[bs,14,14,128]
        x_window3, shortcut3 = self.window3(x_ir3)  # [bs,49,256],[bs,7,7,256]

        o1, o2, o3 = self.attn1(x_window1, q1), self.attn2(x_window2, q2), self.attn3(x_window3, q3)

        # [bs,28,28,64],[bs,14,14,128],[bs,7,7,256]
        o1, o2, o3 = self.ffn1(o1, shortcut1), self.ffn2(o2, shortcut2), self.ffn3(o3, shortcut3)

        # [bs,64,28,28],[bs,128,14,14],[bs,256,7,7]
        o1, o2, o3 = _to_channel_first(o1), _to_channel_first(o2), _to_channel_first(o3)

        return o1, o2, o3





