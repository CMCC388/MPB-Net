import torch
import torch.nn as nn
import torch.nn.functional as F

import network.resnet38d
from util.torchutils import *

class Net(network.resnet38d.Net):
    def __init__(self, num_classes = 3):
        super().__init__()

        self.num_classes = num_classes
        # 添加一个2D Dropout层，用于防止过拟合
        self.dropout7 = torch.nn.Dropout2d(0.5)

        # 添加一个1x1卷积层，用于将4096个特征通道映射到类别数
        self.fc8 = nn.Conv2d(4096, num_classes, 1, bias=False)
        # 使用Xavier初始化权重
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        # 添加一个全连接层用于二分类
        self.fc_binary = nn.Linear(4096, 2)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8
                                    ,self.fc_binary
                                    ]

    def forward(self, x):
        # 调用父类的forward方法
        x = super().forward(x)

        # 应用2D Dropout层
        x = self.dropout7(x)

        # 对特征图进行全局平均池化
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        # 添加二分类的预测
        binary_pred = self.fc_binary(x.view(x.size(0), -1))
        # 使用1x1卷积层生成CAM
        x = self.fc8(x)
        # 将结果展平
        x = x.view(x.size(0), -1)

        return x,binary_pred

    def forward_cams_features(self, x):
        x = super().forward(x)
        feature  = x #b*4096*14*14
        #print(feature.size())
        x =gap2d(feature, keepdims = True)
        x = self.fc8(x)
        x = x.view(-1, self.num_classes) #b*class
        #print(x.size())
        # 使用1x1卷积层生成CAM，并应用ReLU激活函数
        cams = F.conv2d(feature, self.fc8.weight)
        cams = F.relu(cams) #b*num_class*14*14
        #print(cams.size())
        cams = cams/(F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5)
        cams_feature = cams.unsqueeze(2)*feature.unsqueeze(1) # bs*20*2048*32*32
        cams_feature = cams_feature.view(cams_feature.size(0),cams_feature.size(1),cams_feature.size(2),-1)
        cams_feature = torch.mean(cams_feature,-1)
        print(cams_feature.size())
        return cams_feature#,cams
    
    def forward_cam(self, x):
        x = super().forward(x)
        # 使用1x1卷积层生成CAM，并应用ReLU激活函数
        x = F.conv2d(x, self.fc8.weight)
        cams = F.relu(x)
        #cams = cams[0] + cams[1].flip(-1)

        return cams

    
    def forward_recam(self, x, weight):
        x = super().forward(x)
        # 使用1x1卷积层生成CAM，并应用ReLU激活函数
        x = F.conv2d(x, self.fc8.weight*weight)
        cams = F.relu(x)
        #cams = cams[0] + cams[1].flip(-1)

        return cams
    
    def get_parameter_groups(self):
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
    
#（2）空间注意力机制
class spatial_attention(nn.Module):
    # 卷积核大小为7*7
    def __init__(self, kernel_size=7):
        super().__init__()
        
        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2

        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()
    
    # 前向传播
    def forward(self, inputs):
        
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        
        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)
        
        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)

        # 空间权重归一化
        x = self.sigmoid(x)

        # 输入特征图和空间权重相乘
        outputs = inputs * x
        
        return outputs
    
#（3）CBAM注意力机制
class cbam(nn.Module):
    # 初始化，in_channel和ratio=4代表通道注意力机制的输入通道数和第一个全连接下降的通道数
    # kernel_size代表空间注意力机制的卷积核大小
    def __init__(self, in_channel, ratio=4, kernel_size=7):
        super().__init__()
        # 实例化通道注意力机制
        self.channel_attention = channel_attention(in_channel=in_channel, ratio=ratio)
        # 实例化空间注意力机制
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)
    
    # 前向传播
    def forward(self, inputs):
        # 先将输入图像经过通道注意力机制
        x = self.channel_attention(inputs)

        # 然后经过空间注意力机制
        x = self.spatial_attention(x)
        
        return x

