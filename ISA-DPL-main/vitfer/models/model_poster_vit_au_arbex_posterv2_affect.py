import torch
from torch import nn
# from vitfer.models.posterv2.model_irse import IR_50
from vitfer.models.posterv2.model_irse_rasn_affectnet import IR_50
from vitfer.models.two_stream_vit_au_affectnet import HyVisionTransformer
from vitfer.models.posterv2.mobilefacenet import MobileFaceNet
from vitfer.models.trans_fer_affectnet import Local_CNNs
from vitfer.models.posterv2.poster_v2_pure import PosterV2
from vitfer.models.attn import SE_block
from torch.nn import functional as F
from einops import rearrange


class Poster(nn.Module):
    def __init__(self, num_class=7):
        super(Poster, self).__init__()
        # IR-50 with RASN
        pre_weight_ir = './pre_models/backbone_ir50_ms1m_epoch120.pth'
        pre_dict = torch.load(pre_weight_ir, map_location=lambda storage, loc: storage)  # 预训练权重字典不含RASN权重
        self.backbone = IR_50()  # 模型构建包含RASN结构
        model_dict = self.backbone.state_dict()
        pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
        model_dict.update(pre_dict)  # RASN的权重还是初始化
        self.backbone.load_state_dict(model_dict)

        # landmark
        pre_weight_mobilefacenet = './pre_models/mobilefacenet_model_best.pth.tar'
        face_landback_checkpoint = torch.load(pre_weight_mobilefacenet, map_location=lambda storage, loc: storage)
        self.face_landback = MobileFaceNet([112, 112], 136)
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])

        for param in self.face_landback.parameters():
            param.requires_grad = False

        # local_cnns
        self.local_cnns = Local_CNNs()

        # hyp-vit
        self.pyramid_fuse = HyVisionTransformer(in_chans=49, q_chanel=49, embed_dim=512,
                                                depth=8, num_heads=8, mlp_ratio=2.,
                                                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)

        # MLP Head前使用SE可以优化vit
        self.se_block = SE_block(512)

        # vit head
        self.head = nn.Linear(512, num_class)

        # img以及landmark进行cls
        self.cls_img = nn.Parameter(torch.zeros(1, 1, 512))
        self.cls_landmark = nn.Parameter(torch.zeros(1, 1, 512))

        # img进行额外的pos_embed以及pos_drop
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, 512))
        self.pos_drop = nn.Dropout(0.)

        # drop_landmark
        self.drop_landmark = nn.Dropout(0.)

        # poster v2
        self.poster_v2 = PosterV2()
        # self.conv_v2 = nn.Conv1d(49, 49, kernel_size=3, stride=3)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 1)

    def forward(self, inputs):
        # [bs,64,28,28],[bs,128,14,14],[bs,512,7,7]
        x_face1, x_face2, x_face3 = self.face_landback(inputs)
        x_face = self.drop_landmark(x_face3.clone())  # [bs,512,7,7]
        x_face = x_face.reshape(-1, 512, 49).transpose(1, 2)  # [bs,49,512]

        # [bs,64,56,56],[bs,128,28,28],[bs,256,14,14],[bs,512,7,7]
        x_ir1, x_ir2, x_ir3, x = self.backbone(inputs)
        x = self.local_cnns(x)  # [bs,512,7,7]
        x = x.reshape(-1, 512, 49).transpose(1, 2)  # [bs,49,512]

        # poster v2([bs,64,28,28],[bs,128,14,14],[bs,256,7,7])
        o1, o2, o3 = self.poster_v2([x_ir1, x_ir2, x_ir3], [x_face1, x_face2, x_face3])
        # x_o = torch.cat([o1, o2, o3], dim=2)  # [bs,49,512*3]
        # x_o = self.conv_v2(x_o)  # [bs,49,512]

        o1 = self.conv1(o1)  # [bs,128,14,14]
        o2 = o1 + o2  # [bs,128,14,14]
        o2 = self.conv2(o2)  # [bs,256,7,7]
        o3 = o2 + o3  # [bs,256,7,7]
        x_o = self.conv3(o3)  # [bs,512,7,7]
        x_o = x_o.flatten(2).transpose(1, 2)  # [bs,49,512]

        # fuse
        x = x + x_o  # [bs,49,512]

        # 构建cls:[1,1,512]->[bs,1,512]
        cls_img = self.cls_img.expand(x.shape[0], -1, -1)
        cls_landmark = self.cls_landmark.expand(x_face.shape[0], -1, -1)

        # 添加cls:[bs,49,512]->[bs,50,512]
        x = torch.cat((cls_img, x), dim=1)
        x_face = torch.cat((cls_landmark, x_face), dim=1)

        # pos_embed + dropout
        x = self.pos_drop(x + self.pos_embed)  # [bs,50,512]

        # y:[bs,512] au_x_out:[bs,21] x_p:[bs,49,512]
        y, au_x_out, x_p = self.pyramid_fuse(x, x_face)

        # head之前进行注意力机制,维度不变,还是[bs,512]
        feature = self.se_block(y)

        # [bs,512] -> [bs,7/8]
        outs = self.head(feature)

        return outs, feature, au_x_out
