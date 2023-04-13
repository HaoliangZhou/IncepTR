import torch
from torch import nn
from model.Inception_module import Inception
from model.CBAM_module import CBAM
from model.PC_module_womad import VisionTransformer_POS
from torchvision.transforms import Resize
from functools import partial


class DualATTNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(DualATTNet, self).__init__()

        # incep_sca
        self.incep_sca = nn.Sequential(
            Inception(in_channels=3, out_channels=6),
            Inception(in_channels=24, out_channels=12),
            nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1),  # 1*1+BN+ReLU=0.7127,0.6806,0.6814
            # nn.BatchNorm2d(96),  # 1*1+BN=0.7466,0.7190,0.7186
            # nn.ReLU(inplace=True),  # 1*1+ReLU=0.7330,0.6917,0.6926
            CBAM(48),
        )

        # vit_pc
        self.vit_pos = VisionTransformer_POS(img_size=14, patch_size=1, in_chans=3, embed_dim=512, depth=3,
                                             num_heads=4, mlp_ratio=2, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             drop_path_rate=0.3)
        self.resize = Resize([14, 14])

        self.conv1x1_pc = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv1x1_sca = nn.Conv2d(in_channels=48, out_channels=256, kernel_size=1)


        # fc
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=14 * 14 * (256 + 256), out_features=1024),
            # torch.nn.BatchNorm1d(1024),  # 仅BN 70不到
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),  # 仅Drop0.7370
            torch.nn.Linear(in_features=1024, out_features=num_classes),
        )

    def forward(self, x):
        x_sca = self.incep_sca(x)
        x_sca = self.conv1x1_sca(x_sca)

        B = x.shape[0]
        # x = self.vit_pos(x_sca).transpose(1, 2).view(B, 512, 14, 14)
        x_pc = self.vit_pos(self.resize(x)).transpose(1, 2).view(B, 512, 14, 14)
        x_pc = self.conv1x1_pc(x_pc)

        x = torch.cat((x_pc, x_sca), 1)
        x = x.reshape(x.shape[0], -1)  # flatten 变成全连接层的输入
        x = self.fc(x)

        return x_pc, x_sca, x
        # return  x


class DualATTNet_IncepCBAM(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(DualATTNet_IncepCBAM, self).__init__()

        # incep_sca
        self.incep_sca = nn.Sequential(
            Inception(in_channels=3, out_channels=6),
            Inception(in_channels=24, out_channels=12),
            nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1),  # 1*1+BN+ReLU=0.7127,0.6806,0.6814
            # nn.BatchNorm2d(96),  # 1*1+BN=0.7466,0.7190,0.7186
            # nn.ReLU(inplace=True),  # 1*1+ReLU=0.7330,0.6917,0.6926
            CBAM(48),
        )

        self.conv1x1_sca = nn.Conv2d(in_channels=48, out_channels=256, kernel_size=1)

        # fc
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=14 * 14 * (48), out_features=1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=1024, out_features=num_classes),
        )

    def forward(self, x):
        x = self.incep_sca(x)
        # x = self.conv1x1_sca(x)
        x = x.reshape(x.shape[0], -1)  # flatten 变成全连接层的输入
        x = self.fc(x)
        return x


class DualATTNet_ViT(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(DualATTNet_ViT, self).__init__()

        # vit_pc
        self.vit_pos = VisionTransformer_POS(img_size=14,
                                             patch_size=1, in_chans=3, embed_dim=512, depth=3, num_heads=4, mlp_ratio=2,
                                             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             drop_path_rate=0.3)
        self.resize = Resize([14, 14])

        self.conv1x1_pc = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)

        # fc
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=14 * 14 * (256), out_features=1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=1024, out_features=num_classes),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.vit_pos(self.resize(x)).transpose(1, 2).view(B, 512, 14, 14)
        x = self.conv1x1_pc(x)
        x = x.reshape(x.shape[0], -1)  # flatten 变成全连接层的输入
        x = self.fc(x)

        return x

