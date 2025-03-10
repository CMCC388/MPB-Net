from util import torchutils


def get_optimizer(args, model, max_step=None):
    if max_step is None:
        max_step = args.max_iters
    param_groups = model.get_parameter_groups() # 获取模型的参数分组
    # 使用PolyOptimizer创建优化器，该优化器会为不同参数组设置不同的学习率和权重衰减（weight decay）
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    return optimizer # 返回创建的优化器对象

