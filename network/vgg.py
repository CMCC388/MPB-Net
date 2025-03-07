import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import cv2
import numpy as np
import os

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}


# VGG16
class VGG16(nn.Module):
    def __init__(self,num_classes=4096,init_weight=True):
        super(VGG16, self).__init__()
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            # 实例化通道注意力机制1
            #channel_attention(in_channel=64, ratio=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # 实例化通道注意力机制2
            #channel_attention(in_channel=128, ratio=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 实例化通道注意力机制3
            #channel_attention(in_channel=256, ratio=4),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # 实例化通道注意力机制4
            #channel_attention(in_channel=512, ratio=8),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # 实例化通道注意力机制5
            channel_attention(in_channel=512, ratio=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(in_features=7*7*512,out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096,out_features=num_classes)
        )


        # 参数初始化
        if init_weight: # 如果进行参数初始化
            for m in self.modules():  # 对于模型的每一层
                if isinstance(m, nn.Conv2d): # 如果是卷积层
                    # 使用kaiming初始化
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    # 如果bias不为空，固定为0
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Linear):# 如果是线性层
                    # 正态初始化
                    nn.init.normal_(m.weight, 0, 0.01)
                    # bias则固定为0
                    #nn.init.constant_(m.bias, 0)


    def forward(self,x):
        x = self.features(x)
        #x = torch.flatten(x,1)
        #print(x.size()) #[128,512,1,1]
        return x

#（1）通道注意力机制
class channel_attention(nn.Module):
    # ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=4):
        super().__init__()
        
        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        
        # 第一个全连接层, 通道数下降4倍（可以换成1x1的卷积，效果相同）
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel//ratio, bias=False)
        # 第二个全连接层, 恢复通道数（可以换成1x1的卷积，效果相同）
        self.fc2 = nn.Linear(in_features=in_channel//ratio, out_features=in_channel, bias=False)
        
        # relu激活函数
        self.relu = nn.ReLU()

        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
    
    # 前向传播
    def forward(self, inputs):
        b, c, h, w = inputs.shape
        
        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)

        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)
 
        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b,c])
        avg_pool = avg_pool.view([b,c])
 
        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]

        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)
 
        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)
        
        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        #（可以换成1x1的卷积，效果相同）
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)
        
        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool

        # sigmoid函数权值归一化
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b,c,1,1])

        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x
        
        return outputs








class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        
        self.extra_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.fc_extra_3 = nn.Conv2d(512, 64, 1, bias=False)
        self.fc_extra_4 = nn.Conv2d(512, 64, 1, bias=False)
        self.extra_att_module = nn.Conv2d(131, 128, 1, bias=False)
        self.extra_last_conv = nn.Conv2d(512, 1001, 1, bias=False) # 最终输出卷积层
        self._initialize_weights() # 初始化权重
        
        # 初始化一些特定层的权重
        torch.nn.init.xavier_uniform_(self.extra_last_conv.weight)
        torch.nn.init.kaiming_normal_(self.fc_extra_4.weight)
        torch.nn.init.kaiming_normal_(self.fc_extra_3.weight)
        torch.nn.init.xavier_uniform_(self.extra_att_module.weight, gain=4)
        return 

    def forward(self, image):
        out_layer = [23] # 用于指定要提取特征的层的索引
        out_ans = [] # 存储提取的特征
        x = image.clone() # 创建输入数据的副本
        for i in range(len(self.features)):
            x = self.features[i](x) # 通过卷积层传递数据
            if(i in out_layer): # 如果当前层是指定的提取层
                out_ans.append(x) # 将特征添加到列表中
        _, _, h, w = x.size() # 获取最后一个特征图的高度和宽度
        for o in out_ans:
            o = F.interpolate(o, (h, w), mode='bilinear', align_corners=True) # 对提取的特征进行上采样
        image = F.interpolate(image, (h, w), mode='bilinear', align_corners=True) # 对输入图像进行上采样
        f = torch.cat([image, self.fc_extra_3(out_ans[0].detach()), self.fc_extra_4(x.detach())], dim=1) # 连接特征
        x = self.extra_convs(x) # 通过额外的卷积层
        x_att = self.PCM(x, f) # 使用PCM方法计算注意力图
        cam_att = self.extra_last_conv(x_att) # 通过最终卷积层
        cam = self.extra_last_conv(x) # 通过最终卷积层
        # loss = torch.mean(torch.abs(cam_att[:, 1:, :, :] - cam[:, 1:, :, :])) * 0.02

        # 特征图叠加
        self.featmap = cam_att + cam
        logits = F.avg_pool2d(self.featmap[:, :-1], kernel_size=(cam_att.size(2), cam_att.size(3)), padding=0)
        logits = logits.view(-1, 3)

        return self.featmap, logits

    def PCM(self, cam, f):
        n, c, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.extra_att_module(f)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True) # 计算注意力权重
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5) # 归一化权重
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w) # 使用注意力权重融合cam

        return cam_rv # 返回融合后的特征图

    def get_heatmaps(self):
        return self.featmap.clone().detach() # 返回特征图的副本

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i > 13:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2, padding=2)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['D1']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model




# if __name__ == '__main__':
#     model = VGG16()
#     model.forward()