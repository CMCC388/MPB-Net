import os
import torch
import argparse
from torch.backends import cudnn

from util import pyutils

from module.dataloader import get_dataloader
from module.model import get_model
from module.optimizer import get_optimizer
from module.train import *
from torch.optim import lr_scheduler

cudnn.enabled = True
torch.backends.cudnn.benchmark = False

_NUM_CLASSES = {'SD-saliency-900': 3, 'MT': 5}

def get_arguments():
    parser = argparse.ArgumentParser()
    # session
    parser.add_argument("--session", default=
                        #"cls_SD1"
                        #"eps_MTtestrecam+task"
                        #"eps_MT+dice"
                        #"eps_SD+dice"
                        "eps_SD+task0.9"
                        , type=str)

    # data
    parser.add_argument("--data_root", required=False,
                        default='/home/caojun/data/SD-saliency-900/JPEGImages'
                        #default='/home/caojun/data/MT/JPEGImages'
                        , type=str)
    parser.add_argument("--dataset", 
                        default='SD-saliency-900'
                        #default='MT'
                        , type=str)
    parser.add_argument("--saliency_root", type=str,
                        default='/home/caojun/data/SD-saliency-900/SegmentationClassAug'
                        #default='/home/caojun/data/MT/SaliencyImage'
                        )
    parser.add_argument("--train_list", 
                        default="metadata/SD-saliency-900/trainval.txt"
                        #default="metadata/MT/trainval.txt"
                        , type=str)
    parser.add_argument("--infer_list", 
                        default="metadata/SD-saliency-900/trainval.txt"
                        #default="metadata/MT/trainval.txt"
                        , type=str)
    parser.add_argument("--cam_png", default='0.9_tau', type=str)
    parser.add_argument("--thr", default=0.5, type=float)#生成伪标签的阈值
    parser.add_argument("--save_root", default='log')

    #recam设置
    parser.add_argument("--recam", default=True, type=bool)

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--crop_size", default=50, type=int)
    parser.add_argument("--resize_size", default=(224, 224), type=int, nargs='*')

    # network
    parser.add_argument("--network", default="network.resnet38_eps", type=str)
    parser.add_argument("--network_type", default="eps", type=str)
    parser.add_argument("--weights", required=False, type=str,
                        #default='pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params'
                        #default='log/cls_SD/checkpoint_eps_1_0.3357531901245937.pth'
                        #default='log/eps_MTtestrecam+/checkpoint_eps_mt_rc_1_80_0.4248461617528463.pth'

                        default='log/eps_SD+task0.9/checkpoint_eps_mt_rc_1_44_0.2843698863696975.pth'
                        )
    parser.add_argument("--recam_weights", required=False, type=str,
                        default=''
                        #default='log/eps_MTtestrecam+/checkpoint_eps_mt_rc_2_80_0.4248461617528463.pth'
                        #default='log/eps_SDtestrecam/checkpoint_res_112_0.6754942465461451.pth'
                        )
    # optimizer
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--wt_dec", default=5e-7, type=float)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--max_iters", default=20000, type=int)

    # hyper-parameters for EPS
    parser.add_argument("--tau", default=0.9, type=float)#用于阈值化置信区域的阈值
    parser.add_argument("--lam", default=0.5, type=float)#前景图和背景图的混合比率

    args = parser.parse_args()
    #数据集类别
    args.num_classes = _NUM_CLASSES[args.dataset]

    # if 'cls' in args.network:
    #     args.network_type = 'cls'
    # elif 'eps' in args.network:
    #     args.network_type = 'eps'
    # elif 'mt' in args.network:
    #     args.network_type = 'eps+mt'
    # elif 'rc' in args.network:
    #     args.network_type = 'eps+mt+rc'
    # else:
    #     raise Exception('No appropriate model type')

    return args


if __name__ == '__main__':

    # get arguments
    args = get_arguments()

    # set log
    args.log_folder = os.path.join(args.save_root, args.session)
    os.makedirs(args.log_folder, exist_ok=True)
    #输出是否全部存入日志
    pyutils.Logger(os.path.join(args.log_folder, 'log_cls.log'),log_all= True)
    print(vars(args))

    # load dataset
    train_loader = get_dataloader(args)

    max_step = args.max_iters

    # load network and its pre-trained model
    model = get_model(args)

    # set optimizer
    optimizer = get_optimizer(args, model, max_step)



    # train
    #model = torch.nn.DataParallel(model).cuda()
    model.cuda()
    model.train()
    if args.network_type == 'cls':
        train_cls(train_loader, model, optimizer, max_step, args)
    elif args.network_type == 'eps':
    #     train_eps(train_loader, model, optimizer, max_step, args)
    # elif args.network_type == 'eps+mt':
    #     train_eps_mt(train_loader, model, optimizer, max_step, args)
    # elif args.network_type == 'eps+rc':
    #     train_eps_rc(train_loader, model, optimizer, max_step, args)
    # elif args.network_type == 'eps+mt+rc':
        train_eps_mt_rc(train_loader, model, optimizer, max_step, args)
    else:
        raise Exception('No appropriate model type')