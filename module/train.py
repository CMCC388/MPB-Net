import os
import time
import torch
from torch.nn import functional as F

from eps import get_eps_loss,compute_loss
from util import pyutils
from tensorboardX import SummaryWriter
from infer import *
import numpy as np
from PIL import Image
from evaluate import *
from network.recls import *
def train_cls(train_loader, model, optimizer, max_step, args):
    #ReCAM
    recam_predictor = Class_Predictor(3, 4096)
    recam_predictor = recam_predictor.cuda()
    if(args.recam_weight):
        recam_predictor.load_state_dict(torch.load(args.recam_weights))
    recam_predictor.train()
    # 创建一个SummaryWriter，指定日志文件保存路径
    writer = SummaryWriter('logs')
    avg_meter = pyutils.AverageMeter('loss', 'loss_binary', 'loss_multi', 'loss_ce')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_loader)
    step = 0
    max_miou = 0
    for iteration in range(args.max_iters):
        try:
            img_id, img, label = next(loader_iter)
        except:
            loader_iter = iter(train_loader)
            img_id, img, label = next(loader_iter)
        img = img.cuda(non_blocking=True)
        #label = label.type(torch.LongTensor)
        label = label.cuda(non_blocking=True)
        pred,binary_pred = model(img)
        cams_features = model.forward_cams_features(img)

        #类再激活损失
        loss_ce,acc = recam_predictor(cams_features,label)

        # Classification loss多分类标签损失
        loss_multi = F.multilabel_soft_margin_loss(pred, label)
        #二进制交叉熵损失
        #loss_multi = F.binary_cross_entropy_with_logits(pred,label)

        # 计算每行的和
        row_sums = label.sum(dim=1)
        # 创建一个新的标签张量，根据条件输出不同的值
        new_label_tensor = torch.zeros((label.size(0), 2), device='cuda:0')
        new_label_tensor[row_sums == 0] = torch.tensor([1., 0.], device='cuda:0').repeat((row_sums == 0).sum(), 1)
        new_label_tensor[row_sums == 1] = torch.tensor([0., 1.], device='cuda:0').repeat((row_sums == 1).sum(), 1)

        #修改
        # print(pred)
        # print(label)
        # print(binary_pred)
        # print(new_label_tensor)
        
        # Classification loss二分类标签损失
        loss_binary = torch.nn.BCEWithLogitsLoss()(binary_pred,new_label_tensor)
        
        loss = loss_binary + loss_multi +loss_ce
        #loss
        
        #为ReCAM创建优化器
        # 创建优化器，针对网络的参数进行优化
        re_optimizer = torch.optim.SGD(recam_predictor.parameters(), lr=0.01, momentum=0.9)  # 以随机梯度下降（SGD）为例



        avg_meter.add({'loss': loss.item(),
                       'loss_binary': loss_binary.item(),
                       'loss_multi': loss_multi.item(),
                       'loss_ce':loss_ce.item(),
                       })
        # 记录损失到TensorBoard
        writer.add_scalar('Train/Loss', loss, optimizer.global_step)
        writer.add_scalar('Train/Loss_binary', loss_binary, optimizer.global_step)
        writer.add_scalar('Train/Loss_multi', loss_multi, optimizer.global_step)
        writer.add_scalar('Train/Loss_ce', loss_ce, optimizer.global_step)

        optimizer.zero_grad()
        re_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        re_optimizer.step()

        if (optimizer.global_step-1) % 84 == 0:
            timer.update_progress(optimizer.global_step / max_step)

            print('Iter:%5d/%5d' % (iteration, args.max_iters),
                  'Loss:%.4f' % (avg_meter.pop('loss')),
                  'Loss_binary:%.4f' % (avg_meter.pop('loss_binary')),
                  'Loss_multi:%.4f' % (avg_meter.pop('loss_multi')),
                  'Loss_ce:%.4f' % (avg_meter.pop('loss_ce')),
                  'Rem:%s' % (timer.get_est_remain()),
                  'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
        if (optimizer.global_step-1) % 180 == 0:
            #生成伪标签
            print("验证过程开始：")
            evaluate_eps(model,recam_predictor,args)
            miou =compute_miou(args.cam_png,args.saliency_root)
            step = step + 1
            if(miou>max_miou):#如果模型被包装在nn.DataParallel中，那么在这种情况下可能需要使用.module属性
                max_miou = miou
                torch.save(model.state_dict(), os.path.join(args.log_folder, f'checkpoint_cls_{step}_{max_miou}.pth'))
                if(args.recam):
                    torch.save(recam_predictor.state_dict(), os.path.join(args.log_folder, f'checkpoint_res_{step}_{max_miou}.pth'))
        timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_cls_final.pth'))




def train_eps(train_dataloader, model, optimizer, max_step, args):
    #ReCAM
    recam_predictor = Class_Predictor(4, 4096)
    recam_predictor = recam_predictor.cuda()

    # 创建一个SummaryWriter，指定日志文件保存路径
    writer = SummaryWriter('logs')
    avg_meter = pyutils.AverageMeter('loss','loss_multi', 'loss_sal')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
    step = 0
    max_miou = 0
    for iteration in range(args.max_iters):
        try:
            img_id, img, saliency, label, recam_label = next(loader_iter)
        except:
            loader_iter = iter(train_dataloader)
            img_id, img, saliency, label, recam_label = next(loader_iter)
        img = img.cuda(non_blocking=True)
        saliency = saliency.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        recam_label = recam_label.cuda(non_blocking=True)
        pred, cam, binary_pred = model(img)
        cams_features = model.forward_cams_features(img)

        

        # Classification loss
        #多分类损失
        #loss_multi = F.multilabel_soft_margin_loss(pred[:, :-1], label)
        loss_multi = F.binary_cross_entropy_with_logits(pred[:, :-1], label)

        loss_sal, fg_map, bg_map, sal_pred = \
            get_eps_loss(cam, saliency, args.num_classes, label,
                            args.tau, args.lam, intermediate=True)
        loss = loss_multi + loss_sal   #136

        avg_meter.add({'loss': loss.item(),
                        'loss_multi': loss_multi.item(),
                        'loss_sal': loss_sal.item(),
                        })
        
        #loss
        

        # 记录损失到TensorBoard
        writer.add_scalar('Train/Loss', loss, optimizer.global_step)
        writer.add_scalar('Train/Loss_multi', loss_multi, optimizer.global_step)
        writer.add_scalar('Train/Loss_sal', loss_sal, optimizer.global_step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (optimizer.global_step-1) % 1 == 0:
            timer.update_progress(optimizer.global_step / max_step)

            print("Log:"+'Iter:%5d/%5d' % (iteration, args.max_iters),
                    'Loss_Multi:%.4f' % (avg_meter.pop('loss_multi')),
                    'Loss_Sal:%.4f' % (avg_meter.pop('loss_sal')),
                    'Rem:%s' % (timer.get_est_remain()),
                    'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
            
        if (optimizer.global_step-1) % 180 == 0:
            #生成伪标签
            print("验证过程开始：")
            evaluate_eps(model,recam_predictor,args)
            miou =compute_miou(args.cam_png,args.saliency_root)
            step = step + 1
            if(miou>max_miou):#如果模型被包装在nn.DataParallel中，那么在这种情况下可能需要使用.module属性
                max_miou = miou
                torch.save(model.state_dict(), os.path.join(args.log_folder, f'checkpoint_eps_{step}_{max_miou}.pth'))
        timer.reset_stage()
    #torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_eps_final.pth'))

def train_eps_mt(train_dataloader, model, optimizer, max_step, args):
    #ReCAM
    recam_predictor = Class_Predictor(4, 4096)
    recam_predictor = recam_predictor.cuda()
    # 创建一个SummaryWriter，指定日志文件保存路径
    writer = SummaryWriter('logs')
    avg_meter = pyutils.AverageMeter('loss', 'loss_binary','loss_multi', 'loss_sal')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
    step = 0
    max_miou = 0
    for iteration in range(args.max_iters):
        try:
            img_id, img, saliency, label, recam_label = next(loader_iter)
        except:
            loader_iter = iter(train_dataloader)
            img_id, img, saliency, label, recam_label = next(loader_iter)
        img = img.cuda(non_blocking=True)
        saliency = saliency.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        recam_label = recam_label.cuda(non_blocking=True)
        pred, cam, binary_pred = model(img)
        cams_features = model.forward_cams_features(img)

        
        # 计算每行的和
        row_sums = label.sum(dim=1)
        # 创建一个新的标签张量，根据条件输出不同的值
        new_label_tensor = torch.zeros((label.size(0), 2), device='cuda:0')
        new_label_tensor[row_sums == 0] = torch.tensor([1., 0.], device='cuda:0').repeat((row_sums == 0).sum(), 1)
        new_label_tensor[row_sums == 1] = torch.tensor([0., 1.], device='cuda:0').repeat((row_sums == 1).sum(), 1)

        # Classification loss
        #二分类损失
        loss_binary = torch.nn.BCEWithLogitsLoss()(binary_pred,new_label_tensor)
        #多分类损失
        #loss_multi = F.multilabel_soft_margin_loss(pred[:, :-1], label)
        loss_multi = F.binary_cross_entropy_with_logits(pred[:, :-1], label)

        loss_sal, fg_map, bg_map, sal_pred = \
            get_eps_loss(cam, saliency, args.num_classes, label,
                            args.tau, args.lam, intermediate=True)
        loss = 0.1*loss_binary + 0.3*loss_multi + 0.6*loss_sal    #136

        avg_meter.add({'loss': loss.item(),
                        'loss_binary': loss_binary.item(),
                        'loss_multi': loss_multi.item(),
                        'loss_sal': loss_sal.item(),
                        })
        
        #loss
        

        # 记录损失到TensorBoard
        writer.add_scalar('Train/Loss', loss, optimizer.global_step)
        writer.add_scalar('Train/Loss_binary', loss_binary, optimizer.global_step)
        writer.add_scalar('Train/Loss_multi', loss_multi, optimizer.global_step)
        writer.add_scalar('Train/Loss_sal', loss_sal, optimizer.global_step)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


        if (optimizer.global_step-1) % 1 == 0:
            timer.update_progress(optimizer.global_step / max_step)

            print("Log:"+'Iter:%5d/%5d' % (iteration, args.max_iters),
                    'Loss_Binary:%.4f' % (avg_meter.pop('loss_binary')),
                    'Loss_Multi:%.4f' % (avg_meter.pop('loss_multi')),
                    'Loss_Sal:%.4f' % (avg_meter.pop('loss_sal')),
                    'Rem:%s' % (timer.get_est_remain()),
                    'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
            
        if (optimizer.global_step-1) % 180 == 0:
            #生成伪标签
            print("验证过程开始：")
            evaluate_eps(model,recam_predictor,args)
            miou =compute_miou(args.cam_png,args.saliency_root)
            step = step + 1
            if(miou>max_miou):#如果模型被包装在nn.DataParallel中，那么在这种情况下可能需要使用.module属性
                max_miou = miou
                torch.save(model.state_dict(), os.path.join(args.log_folder, f'checkpoint_eps_mt_1_{step}_{max_miou}.pth'))
                
        timer.reset_stage()
    #torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_eps_final.pth'))

def train_eps_rc(train_dataloader, model, optimizer, max_step, args):
    #ReCAM
    recam_predictor = Class_Predictor(6, 4096)
    recam_predictor = recam_predictor.cuda()
    if(args.recam_weight):
        recam_predictor.load_state_dict(torch.load(args.recam_weights))
    recam_predictor.train()
    # 创建一个SummaryWriter，指定日志文件保存路径
    writer = SummaryWriter('logs')
    avg_meter = pyutils.AverageMeter('loss','loss_multi', 'loss_sal', 'loss_ce')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
    step = 0
    max_miou = 0
    for iteration in range(args.max_iters):
        try:
            img_id, img, saliency, label, recam_label = next(loader_iter)
        except:
            loader_iter = iter(train_dataloader)
            img_id, img, saliency, label, recam_label = next(loader_iter)
        img = img.cuda(non_blocking=True)
        saliency = saliency.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        recam_label = recam_label.cuda(non_blocking=True)
        pred, cam, binary_pred = model(img)
        cams_features = model.forward_cams_features(img)

        #类再激活损失
        loss_ce,acc = recam_predictor(cams_features,recam_label)

        # Classification loss
        #多分类损失
        #loss_multi = F.multilabel_soft_margin_loss(pred[:, :-1], label)
        loss_multi = F.binary_cross_entropy_with_logits(pred[:, :-1], label)

        loss_sal, fg_map, bg_map, sal_pred = \
            get_eps_loss(cam, saliency, args.num_classes, label,
                            args.tau, args.lam, intermediate=True)
        loss = 0.3*loss_multi + 0.6*loss_sal + 0.1*loss_ce   #136

        avg_meter.add({'loss': loss.item(),
                        'loss_multi': loss_multi.item(),
                        'loss_sal': loss_sal.item(),
                        'loss_ce': loss_ce.item()})
        
        #loss
        
        #为ReCAM创建优化器
        # 创建优化器，针对网络的参数进行优化
        re_optimizer = torch.optim.SGD(recam_predictor.parameters(), lr=0.01, momentum=0.9)  # 以随机梯度下降（SGD）为例

        # 记录损失到TensorBoard
        writer.add_scalar('Train/Loss', loss, optimizer.global_step)
        writer.add_scalar('Train/Loss_multi', loss_multi, optimizer.global_step)
        writer.add_scalar('Train/Loss_sal', loss_sal, optimizer.global_step)
        writer.add_scalar('Train/Loss_ce', loss_ce, optimizer.global_step)

        optimizer.zero_grad()
        re_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        re_optimizer.step()

        if (optimizer.global_step-1) % 1 == 0:
            timer.update_progress(optimizer.global_step / max_step)

            print("Log:"+'Iter:%5d/%5d' % (iteration, args.max_iters),
                    'Loss_Multi:%.4f' % (avg_meter.pop('loss_multi')),
                    'Loss_Sal:%.4f' % (avg_meter.pop('loss_sal')),
                    'Loss_Ce:%.4f' % (avg_meter.pop('loss_ce')),
                    'Rem:%s' % (timer.get_est_remain()),
                    'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
            
        if (optimizer.global_step-1) % 180 == 0:
            #生成伪标签
            print("验证过程开始：")
            evaluate_eps(model,recam_predictor,args)
            miou =compute_miou(args.cam_png,args.saliency_root)
            step = step + 1
            if(miou>max_miou):#如果模型被包装在nn.DataParallel中，那么在这种情况下可能需要使用.module属性
                max_miou = miou
                torch.save(model.state_dict(), os.path.join(args.log_folder, f'checkpoint_eps_rc_1_{step}_{max_miou}.pth'))
                torch.save(recam_predictor.state_dict(), os.path.join(args.log_folder, f'checkpoint_eps_rc_2_{step}_{max_miou}.pth'))
        timer.reset_stage()
    #torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_eps_final.pth'))

def train_eps_mt_rc(train_dataloader, model, optimizer, max_step, args):
    #ReCAM
    recam_predictor = Class_Predictor(4, 4096)
    recam_predictor = recam_predictor.cuda()
    if(args.recam_weights):
        recam_predictor.load_state_dict(torch.load(args.recam_weights))
    recam_predictor.train()
    # 创建一个SummaryWriter，指定日志文件保存路径
    writer = SummaryWriter('logs')
    avg_meter = pyutils.AverageMeter('loss', 'loss_binary','loss_multi', 'loss_sal', 'loss_ce')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
    step = 0
    max_miou = 0
    for iteration in range(args.max_iters):
        try:
            img_id, img, saliency, label, recam_label = next(loader_iter)
        except:
            loader_iter = iter(train_dataloader)
            img_id, img, saliency, label, recam_label = next(loader_iter)
        img = img.cuda(non_blocking=True)
        saliency = saliency.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        recam_label = recam_label.cuda(non_blocking=True)
        pred, cam, binary_pred = model(img)
        cams_features = model.forward_cams_features(img)

        #类再激活损失
        loss_ce,acc = recam_predictor(cams_features,recam_label)
        # 计算每行的和
        row_sums = label.sum(dim=1)
        # 创建一个新的标签张量，根据条件输出不同的值
        new_label_tensor = torch.zeros((label.size(0), 2), device='cuda:0')
        new_label_tensor[row_sums == 0] = torch.tensor([1., 0.], device='cuda:0').repeat((row_sums == 0).sum(), 1)
        new_label_tensor[row_sums == 1] = torch.tensor([0., 1.], device='cuda:0').repeat((row_sums == 1).sum(), 1)

        # Classification loss
        #二分类损失
        loss_binary = torch.nn.BCEWithLogitsLoss()(binary_pred,new_label_tensor)
        #多分类损失
        #loss_multi = F.multilabel_soft_margin_loss(pred[:, :-1], label)
        loss_multi = F.binary_cross_entropy_with_logits(pred[:, :-1], label)

        loss_sal, fg_map, bg_map, sal_pred = \
            get_eps_loss(cam, saliency, args.num_classes, label,
                            args.tau, args.lam, intermediate=True)
        # loss_sal, fg_map, bg_map, sal_pred = \
        #     compute_loss(cam, saliency, args.num_classes, label,
        #                     args.tau, args.lam, intermediate=True)
        a = 0.5
        b = 1
        c = 1
        d = 0
        # if iteration>30000:
        #     a = 0.5
        # if iteration>60000:
        #     c = 1
        if iteration>10000:
            d = 1

        loss = a*loss_binary + b*loss_multi + c*loss_sal + d*loss_ce   #136

        avg_meter.add({'loss': loss.item(),
                        'loss_binary': loss_binary.item(),
                        'loss_multi': loss_multi.item(),
                        'loss_sal': loss_sal.item(),
                        'loss_ce': loss_ce.item()})
        
        #loss
        
        #为ReCAM创建优化器
        # 创建优化器，针对网络的参数进行优化
        re_optimizer = torch.optim.SGD(recam_predictor.parameters(), lr=0.01, momentum=0.9)  # 以随机梯度下降（SGD）为例

        # 记录损失到TensorBoard
        writer.add_scalar('Train/Loss', loss, optimizer.global_step)
        writer.add_scalar('Train/Loss_binary', loss_binary, optimizer.global_step)
        writer.add_scalar('Train/Loss_multi', loss_multi, optimizer.global_step)
        writer.add_scalar('Train/Loss_sal', loss_sal, optimizer.global_step)
        writer.add_scalar('Train/Loss_ce', loss_ce, optimizer.global_step)

        optimizer.zero_grad()
        re_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        re_optimizer.step()

        if (optimizer.global_step-1) % 1 == 0:
            timer.update_progress(optimizer.global_step / max_step)

            print("Log:"+'Iter:%5d/%5d' % (iteration, args.max_iters),
                    'Loss_Binary:%.4f' % (avg_meter.pop('loss_binary')),
                    'Loss_Multi:%.4f' % (avg_meter.pop('loss_multi')),
                    'Loss_Sal:%.4f' % (avg_meter.pop('loss_sal')),
                    'Loss_Ce:%.4f' % (avg_meter.pop('loss_ce')),
                    'Rem:%s' % (timer.get_est_remain()),
                    'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
        if iteration>1000  :  
            if (optimizer.global_step-1) % 180 == 0:
                #生成伪标签
                print("验证过程开始：")
                evaluate_eps(model,recam_predictor,args)
                miou =compute_miou(args.cam_png,args.saliency_root)
                step = step + 1
                if(miou>max_miou):#如果模型被包装在nn.DataParallel中，那么在这种情况下可能需要使用.module属性
                    max_miou = miou
                    torch.save(model.state_dict(), os.path.join(args.log_folder, f'checkpoint_eps_mt_rc_1_{step}_{max_miou}.pth'))
                    if(args.recam):
                        torch.save(recam_predictor.state_dict(), os.path.join(args.log_folder, f'checkpoint_eps_mt_rc_2_{step}_{max_miou}.pth'))
            timer.reset_stage()
    #torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_eps_final.pth'))

def print_progress_bar(iteration, total, bar_length=50):
    progress = (iteration / total)
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    print(f'[{arrow}{spaces}] {int(progress * 100)}%', end='\r')

def evaluate_eps(model, reclsmodel, args):
    if args.dataset == 'voc12':
        args.num_classes = 20
    elif args.dataset == 'coco':
        args.num_classes = 80
    elif args.dataset == 'SD-saliency-900':
        args.num_classes = 3
    elif args.dataset == 'MT':
        args.num_classes = 5
    else:
        raise Exception('Error')

    # model information
    if 'cls' in args.network:
        args.network_type = 'cls'
        args.model_num_classes = args.num_classes
    elif 'eps' in args.network:
        args.network_type = 'eps'
        args.model_num_classes = args.num_classes + 1
    else:
        raise Exception('No appropriate model type')
    
    gpu = 1
    
    image_ids = load_img_id_list(args.infer_list)
    label_list = load_img_label_list_from_npy(image_ids, args.dataset)
    n_total_images = len(image_ids)
    print('n_total_images:', n_total_images)
    assert len(image_ids) == len(label_list)
    model.eval()
    reclsmodel.eval()
    torch.no_grad()

    for i,(img_id, label) in enumerate(zip(image_ids, label_list)):

        # load image
        # 获取文件名前缀作为类别标识
        class_prefix = img_id.split("_")[0]
        class_prefix = int(class_prefix)
        if(class_prefix>0):#无缺陷图片不生成CAM
            img_path = os.path.join(args.data_root, str(img_id) + '.jpg') # 构建图像文件路径
            img = Image.open(img_path).convert('RGB') # 打开图像并将其转换为RGB模式
            org_img = np.asarray(img) # 将图像转换为NumPy数组

            # infer cam_list
            cam_list = predict_cam(model, img, label, reclsmodel,args) # 使用模型生成CAM列表

            if args.network_type == 'cls':
                sum_cam = np.sum(cam_list, axis=0)  # 如果网络类型为'cls'，则将CAM列表求和
            elif args.network_type == 'eps':
                cam_np = np.array(cam_list,dtype=object)
                cam_fg = cam_np[:, 0]
                sum_cam = np.sum(cam_fg, axis=0)  # 如果网络类型为'eps'，则将前景CAM列表求和
            else:
                raise Exception('No appropriate model type')
            norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)  # 归一化CAM

            cam_dict = {}
            for j in range(args.num_classes):
                if label[j] > 1e-5:
                    cam_dict[j] = norm_cam[j]  # 为每个类别创建CAM字典
            h, w = list(cam_dict.values())[0].shape

            # 在创建颜色映射时，将其形状扩展为与tensor相同
            color_map = create_color_map(args.num_classes)
            tensor = np.zeros((args.num_classes + 1, h, w), np.float32)  # 创建一个包含类别数加一的零张量
            for key in cam_dict.keys():
                tensor[key + 1] = cam_dict[key]  # 将归一化的CAM填充到张量中
                #tensor[key + 1] = color_map[key]
            
            # #SD方案
            # # 创建分割掩码，将不同类别映射到不同的颜色
            # pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
            # for key in cam_dict.keys():
            #     if key == 0:
            #         tensor[0, :, :] = 0.59  # 在第一个通道中添加阈值
            #     elif key == 1:
            #         tensor[0, :, :] = 0.57  # 在第一个通道中添加阈值
            #     elif key == 2:
            #         tensor[0, :, :] = 0.451  # 在第一个通道中添加阈值
            #     pred = np.argmax(tensor, axis=0).astype(np.uint8)  # 根据张量的最大值确定最终的预测
            #     pred_colored[pred==key+1] = color_map[key+1]  # 使用颜色映射为每个类别上色
            #通用方案
            tensor[0, :, :] = args.thr  # 在第一个通道中添加阈值
            pred = np.argmax(tensor, axis=0).astype(np.uint8)  # 根据张量的最大值确定最终的预测

            # 创建分割掩码，将不同类别映射到不同的颜色
            pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
            for key in cam_dict.keys():
                pred_colored[pred==key+1] = color_map[key+1]  # 使用颜色映射为每个类别上色

            # #MT方案
            # pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
            # for key in cam_dict.keys():
            #     if key == 0:
            #         tensor[0, :, :] = 0.91 # 在第一个通道中添加阈值
            #     elif key == 1:
            #         tensor[0, :, :] = 0.35  # 在第一个通道中添加阈值
            #     elif key == 2:
            #         tensor[0, :, :] = 0.35  # 在第一个通道中添加阈值
            #     elif key == 3:
            #         tensor[0, :, :] = 0.35  # 在第一个通道中添加阈值
            #     elif key == 4:
            #         tensor[0, :, :] = 0.37  # 在第一个通道中添加阈值
            #     pred = np.argmax(tensor, axis=0).astype(np.uint8)  # 根据张量的最大值确定最终的预测
            #     pred_colored[pred==key+1] = color_map[key+1]  # 使用颜色映射为每个类别上色


            # save cam
            if args.cam_png is not None:
                imageio.imwrite(os.path.join(args.cam_png, str(img_id) + '.png'), pred_colored) # 保存预测结果为.png图像
            # 打印进度条
            print_progress_bar(i + 1, n_total_images)
    print()




