import torch
from torch import nn


class SelfAttn(nn.Module):
    def __init__(self, n_anchors=1, n_classes=8, size_emb=768, nhead=1):
        super().__init__()
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        size_out = n_classes * n_anchors
        self.Q = nn.Linear(size_emb, size_out)
        self.K = nn.Linear(size_emb, size_out)
        self.V = nn.Linear(size_emb, size_out)
        self.attn_fn = nn.MultiheadAttention(size_out, num_heads=nhead)

    def forward(self, x):
        # q,k,v需要三维,因为此时qkv代表cls token,序列长度为1
        q = self.Q(x).unsqueeze(dim=0)  # [1,bs,7]
        k = self.K(x).unsqueeze(dim=0)  # [1,bs,7]
        v = self.V(x).unsqueeze(dim=0)  # [1,bs,7]
        attn_scores, _ = self.attn_fn(q, k, v)
        attn_scores = torch.softmax(attn_scores, -1)

        return attn_scores.view(-1, self.n_classes, self.n_anchors).sum(-1)


class ClassificationHead(nn.Module):
    def __init__(self, size_in=512, size_out=7, size_hidden=[256, 128, 64], dropout=0.5, batch_norm=False):
        super().__init__()

        self.size_in = size_in
        self.size_out = size_out
        self.size_hidden = size_hidden
        self.dropout = dropout
        self.batch_norm = batch_norm

        net = []
        for h in self.size_hidden:
            net.append(nn.Linear(size_in, h))
            net.append(nn.ReLU())
            if self.batch_norm:
                net.append(nn.BatchNorm1d(h))
            net.append(nn.Dropout(self.dropout))
            size_in = h

        net.append(nn.Linear(h, self.size_out))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x


class Anchors(nn.Module):
    def __init__(self, size_emb=512, n_classes=7, n_anchors=10):
        super().__init__()
        self.size_emb = size_emb
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        anchors = torch.zeros((self.n_classes, self.n_anchors, self.size_emb))
        self.anchors = nn.Parameter(anchors)

    # x:[bs,embedding]
    def forward(self, x):
        x = x.view(x.shape[0], 1, 1, x.shape[1])  # [batch, 1, 1, emb]
        anchors = self.anchors.unsqueeze(0)  # [1, classes, anchors, emb]
        distances = (anchors - x) ** 2  # [batch, classes, anchors, emb]
        distances = distances.sum(-1)  # [batch, classes, anchors]
        distances = torch.sqrt(distances)  # [batch, classes, anchors]

        return distances

    def get_anchors(self):
        return self.anchors
