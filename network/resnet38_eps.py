import torch
import torch.nn as nn
import torch.nn.functional as F

import network.resnet38d
from torchcam.methods import GradCAM
from util.torchutils import *
class Net(network.resnet38d.Net):
    def __init__(self, num_classes = 4):
        super().__init__()
        self.num_classes = num_classes

        # 添加一个1x1卷积层，用于将4096个特征通道映射到类别数
        self.fc8 = nn.Conv2d(4096, num_classes, 1, bias=False)


        # 使用Xavier初始化权重
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        # 添加一个全连接层用于二分类
        self.fc_binary = nn.Linear(4096, 2)
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8
                                    , self.fc_binary
                                    ]


    def forward(self, x):
        # 调用父类的forward方法
        x = super().forward(x)


        # 对特征图进行全局平均池化
        x1 = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        # 添加二分类的预测
        binary_pred = self.fc_binary(x1.view(x1.size(0), -1))
        #features = x
        # 计算CAM（类激活图）
        cam = self.fc8(x)

        _, _, h, w = cam.size()
        # 对CAM进行全局平均池化
        pred = F.avg_pool2d(cam, kernel_size=(h, w), padding=0)

        # 将结果展平
        pred = pred.view(pred.size(0), -1)


        return pred, cam ,binary_pred
        #return features
        # return pred
    def forward_features(self, x):
        x = super().forward(x)
        x = torch.nn.AdaptiveAvgPool2d((1, 1))(x)
        features = x

        return features
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
        #print(cams_feature.size())
        return cams_feature#,cams
    
    def forward_cam(self, x):
        x = super().forward(x)
        # 使用1x1卷积层生成CAM，并应用ReLU激活函数
        cams = F.conv2d(x, self.fc8.weight)
        #cams = F.relu(cams)
        #cams = cams[0] + cams[1].flip(-1)

        return cams

    def forward_recam(self, x, weight):
        x = super().forward(x)
        #print(weight.size()) #[4,4096,1,1]
        #print(self.fc8.weight.size()) #[4,4096,1,1]
        weight = F.normalize(weight, p=2, dim=0)
        # 使用1x1卷积层生成CAM，并应用ReLU激活函数
        cams = F.conv2d(x, self.fc8.weight*weight)
        #cams = F.relu(cams)
        #cams = cams[0] + cams[1].flip(-1)

        return cams
    def forward_recam1(self, x, weight):
        x = super().forward(x)
        #print(weight.size()) #[4,4096,1,1]
        #print(self.fc8.weight.size()) #[4,4096,1,1]
        weight = F.normalize(weight, p=2, dim=0)
        # 使用1x1卷积层生成CAM，并应用ReLU激活函数
        cams = F.conv2d(x, self.fc8.weight+weight)
        #cams = F.relu(cams)
        #cams = cams[0] + cams[1].flip(-1)

        return cams
    def forward_recam2(self, x, weight):
        x = super().forward(x)
        #print(weight.size()) #[4,4096,1,1]
        #print(self.fc8.weight.size()) #[4,4096,1,1]
        weight = F.normalize(weight, p=2, dim=0)
        # 使用1x1卷积层生成CAM，并应用ReLU激活函数
        cams = F.conv2d(x, weight)
        #cams = F.relu(cams)
        #cams = cams[0] + cams[1].flip(-1)

        return cams
    
    
    def forward_gradcam(self,x):
        # 注册钩子
        def forward_hook(module, input, output):
            setattr(module, "_value_hook", output)

        target_layer = getattr(self, self.target_layer)
        hook = target_layer.register_forward_hook(forward_hook)
        scores = self.forward(x)
        print('输出scores:')
        print(scores.size())
        cam_extractor = GradCAM(model=self,target_layer=self.target_layer)
        class_idx = scores.squeeze(0).argmax().item()
        print('输出class_idx:')
        print(class_idx)
        gradcam = cam_extractor(class_idx=class_idx,scores=scores)
        print('输出成功')
        # 移除钩子
        hook.remove()
        return gradcam

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