import torch
import torch.nn as nn
import torch.nn.functional as F

import network.vgg


class Net(network.vgg.VGG16):
    def __init__(self,num_classes):
        super().__init__()
        # 添加一个1x1卷积层，用于将512个特征通道映射到类别数
        self.fc8 = nn.Conv2d(512, num_classes, 1, bias=False)
        # 使用Xavier初始化权重
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.from_scratch_layers = [self.fc8]

    def forward(self, x):
        # 调用父类的forward方法
        x = super().forward(x)
        # 计算CAM（类激活图）
        cam = self.fc8(x)

        _, _, h, w = cam.size()
        # 对CAM进行全局平均池化
        pred = F.avg_pool2d(cam, kernel_size=(h, w), padding=0)
        # 将结果展平
        pred = pred.view(pred.size(0), -1)
        return pred, cam

    def forward_cam(self, x):
        x = super().forward(x)
        cam = self.fc8(x)

        return cam

    def get_parameter_groups(self):
        # 将网络的参数分成四个组：不参与训练的权重、需要训练的权重、不参与训练的偏置、需要训练的偏置
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups