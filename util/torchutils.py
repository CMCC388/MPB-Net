import torch

#
class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)  # 继承自 torch.optim.SGD 并调用其初始化方法

        self.global_step = 0 # 初始化全局步数为 0
        self.max_step = max_step # 设置总的最大步数
        self.momentum = momentum # 设置动量，默认为 0.9

        self.__initial_lr = [group['lr'] for group in self.param_groups] # 获取每个参数组的初始学习率

    def step(self, closure=None):
        # 如果当前全局步数小于最大步数，执行学习率调整
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum # 计算学习率的多项式衰减因子

            for i in range(len(self.param_groups)):
                 # 根据多项式衰减因子，调整每个参数组的学习率
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure) # 调用父类 torch.optim.SGD 的 step 方法来执行参数更新

        self.global_step += 1 # 增加全局步数

def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out

def gap2d_pos(x, keepdims=False):
    out = torch.sum(x.view(x.size(0), x.size(1), -1), -1) / (torch.sum(x>0)+1e-12)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out

def gsp2d(x, keepdims=False):
    out = torch.sum(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out

