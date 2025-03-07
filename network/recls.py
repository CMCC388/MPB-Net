import torch
from torch import nn

import torch.nn.functional as F


class Class_Predictor(nn.Module):
    def __init__(self, num_classes, representation_size):
        super(Class_Predictor, self).__init__()
        # 设置分类数和特征维度
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(representation_size, num_classes, 1, bias=False)
        # # 使用Xavier初始化权重
        # torch.nn.init.xavier_uniform_(self.classifier.weight)
        # 将权重初始化为全为一
        torch.nn.init.ones_(self.classifier.weight)

    def forward(self, x, label):
        #print(x)
        batch_size = x.shape[0] # 获取批次大小
        x = x.reshape(batch_size,self.num_classes,-1) # bs*20*2048 # 将特征张量重塑为 batch_size * num_classes * feature_size
        mask = label>0 # bs*20  # 根据标签创建掩码
        # print(mask.size())
        # print(mask)
        # 根据掩码提取特征
        feature_list = [x[i][mask[i]] for i in range(batch_size)] # bs*n*2048

        # 使用分类器对特征进行分类预测
        prediction = [self.classifier(y.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) for y in feature_list]
        
        # 获取非零标签的索引
        labels = [torch.nonzero(label[i]).squeeze(1) for i in range(label.shape[0])]

        loss = 0 # 初始化损失
        acc = 0 # 初始化正确预测数
        num = 0 # 初始化总数
        
        # 计算损失和准确率
        for logit,label in zip(prediction, labels):
            if label.shape[0] == 0:
                continue
            loss_ce= F.cross_entropy(logit, label) # 计算交叉熵损失
            loss += loss_ce # 累加损失
            acc += (logit.argmax(dim=1)==label.view(-1)).sum().float() # 计算正确预测数
            num += label.size(0) # 计算总数
            
        return loss/batch_size, acc/num # 返回平均损失和准确率