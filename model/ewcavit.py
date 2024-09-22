import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from CBAM import CBAM


class ResNet(torch.nn.Module):
    def __init__(self, useCBAM=False):
        super(ResNet, self).__init__()
        self.model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
        self.features = torch.nn.Sequential(*list(self.model.children())[:5])
        self.cbam = CBAM(256)
        self.useCBAM = useCBAM
        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        for name, param in self.features.named_parameters():
            param.requires_grad = False
        x = self.features(x)
        if self.useCBAM:
            x_cbam = self.cbam(x)
            x_out = self.deconv(x_cbam)
        else:
            x_out = self.deconv(x)
        return x_out


resnet = ResNet()


class VGGNet(torch.nn.Module):
    def __init__(self, useCBAM=False):
        super(VGGNet, self).__init__()
        self.model = models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1")
        self.features = torch.nn.Sequential(*list(self.model.features.children())[:17])
        self.cbam = CBAM(256)
        self.useCBAM = useCBAM
        self.deconv = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear"),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        for name, param in self.features.named_parameters():
            param.requires_grad = False
        x = self.features(x)
        if self.useCBAM:
            x_cbam = self.cbam(x)
            x_out = self.deconv(x_cbam)
        else:
            x_out = self.deconv(x)
        return x_out


class MobileNet(torch.nn.Module):
    def __init__(self, useCBAM=False):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v3_large(
            weights="MobileNet_V3_Large_Weights.IMAGENET1K_V1"
        )
        self.features = torch.nn.Sequential(*list(self.model.features.children())[:2])
        self.cbam = CBAM(16)
        self.useCBAM = useCBAM
        self.deconv = torch.nn.Sequential(
            nn.Dropout2d(0.5),
            torch.nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        for name, param in self.features.named_parameters():
            param.requires_grad = False
        x = self.features(x)
        if self.useCBAM:
            x_cbam = self.cbam(x)
            x_out = self.deconv(x_cbam)
        else:
            x_out = self.deconv(x)
        return x_out


class SwinTransformer(torch.nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        self.backbone = models.swin_t(weights="Swin_T_Weights.IMAGENET1K_V1")
        self.features = create_feature_extractor(
            self.backbone, return_nodes=["flatten"]
        )

    def forward(self, x):
        for name, param in self.features.named_parameters():
            param.requires_grad = False
        return self.features(x)


class Head(torch.nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(768, 2),
        )

    def forward(self, x):
        return self.mlp_head(x)


class EWCAT(torch.nn.Module):
    def __init__(self):
        super(EWCAT, self).__init__()
        self.vggnet = VGGNet(useCBAM=False)
        self.resnet = ResNet(useCBAM=False)
        self.mobilenet = MobileNet(useCBAM=False)
        # self.dense = DenseNet(useCBAM = False)
        self.swin_transformer = SwinTransformer()
        self.head = Head()
        self.cbam = CBAM(3)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x, features=False):
        f_vgg = self.vggnet(x)
        f_res = self.resnet(x)
        f_mob = self.mobilenet(x)

        # v3.21
        f_vgg = self.bn(f_vgg)
        f_res = self.bn(f_res)
        f_mob = self.bn(f_mob)
        f_vgg = self.cbam(f_vgg)
        f_res = self.cbam(f_res)
        f_mob = self.cbam(f_mob)
        kV, kR, kD = 1 / 3, 1 / 3, 1 / 3  # V3
        inte = kV * f_vgg + kR * f_res + kD * f_mob
        tran_out = self.swin_transformer(inte)
        if features:
            return tran_out["flatten"]
        else:
            return self.head(tran_out["flatten"])
