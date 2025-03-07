import os
import time
import imageio
import argparse
import importlib
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F
from torch.multiprocessing import Process


from evaluate import compute_miou
from util import imutils, pyutils
from util.imutils import HWC_to_CHW
from network.resnet38d import Normalize
from metadata.dataset import load_img_id_list, load_img_label_list_from_npy
#from torchcam.methods import GradCAM
from network.recls import *

start = time.time()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet38_eps", type=str)
    parser.add_argument("--weights", required=False,default='log/eps_SD+dice/checkpoint_eps_mt_rc_1_33_0.7269697634127853.pth',type=str)
    parser.add_argument("--recam_weights", required=False,default='log/eps_SD+dice/checkpoint_eps_mt_rc_2_33_0.7269697634127853.pth',type=str)
    #recam设置
    parser.add_argument("--recam", default=True, type=bool)
    parser.add_argument("--n_gpus", type=int, default=2)
    parser.add_argument("--infer_list", 
                        default="metadata/SD-saliency-900/trainval_1.txt"
                        #default="metadata/MT copy/trainval.txt"
                        , type=str)
    parser.add_argument("--n_total_processes", default=1, type=int)
    parser.add_argument("--img_root", 
                        default='/home/caojun/data/SD-saliency-900/JPEGImages'
                        #default='/home/caojun/data/MT/JPEGImages'
                        , type=str)
    parser.add_argument("--crf", default=None, type=str)
    parser.add_argument("--crf_alpha", nargs='*', type=int)
    parser.add_argument("--crf_t", nargs='*', type=int)
    parser.add_argument("--cam_npy", default=None, type=str)
    parser.add_argument("--pred_cam", default='testww', type=str)
    parser.add_argument("--true_cam", type=str,
                        default='/home/caojun/data/SD-saliency-900/SegmentationClassAug'
                        #default='/home/caojun/data/MT/SaliencyImage'
                        )    
    parser.add_argument("--thr", default=0.5, type=float)
    parser.add_argument("--dataset", 
                        #default='MT'
                        default='SD-saliency-900'
                        , type=str)
    args = parser.parse_args()

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

    # save path
    args.save_type = list()
    if args.cam_npy is not None:
        os.makedirs(args.cam_npy, exist_ok=True)
        args.save_type.append(args.cam_npy)
    if args.pred_cam is not None:
        os.makedirs(args.pred_cam, exist_ok=True)
        args.save_type.append(args.pred_cam)
    if args.crf:
        args.crf_list = list()
        for t in args.crf_t:
            for alpha in args.crf_alpha:
                crf_folder = os.path.join(args.crf, 'crf_{}_{}'.format(t, alpha))
                os.makedirs(crf_folder, exist_ok=True)
                args.crf_list.append((crf_folder, t, alpha))
                args.save_type.append(args.crf_folder)


    return args


def preprocess(image, scale_list, transform):
    img_size = image.size
    num_scales = len(scale_list)
    multi_scale_image_list = list()
    multi_scale_flipped_image_list = list()

    # insert multi-scale images
    for s in scale_list:
        target_size = (round(img_size[0] * s), round(img_size[1] * s))
        scaled_image = image.resize(target_size, resample=Image.CUBIC)
        multi_scale_image_list.append(scaled_image)
    # transform the multi-scaled image
    for i in range(num_scales):
        multi_scale_image_list[i] = transform(multi_scale_image_list[i])
    # augment the flipped image
    for i in range(num_scales):
        multi_scale_flipped_image_list.append(multi_scale_image_list[i])
        multi_scale_flipped_image_list.append(np.flip(multi_scale_image_list[i], -1).copy())
    return multi_scale_flipped_image_list


def predict_cam(model, image, label, reclsmodel, args):
    scales = (0.5, 1.0, 1.5, 2.0)
    normalize = Normalize()
    transform = torchvision.transforms.Compose([np.asarray, normalize, HWC_to_CHW])
    original_image_size = np.asarray(image).shape[:2]
    # preprocess image
    multi_scale_flipped_image_list = preprocess(image, scales, transform)

    cam_list = list()
    #model.eval()
    for i, image in enumerate(multi_scale_flipped_image_list):
        with torch.no_grad():
            image = torch.from_numpy(image).unsqueeze(0)
            image = image.cuda() #删除gpu
            #因recam修改
            if(args.recam):
                cam = model.forward_recam(image, reclsmodel.classifier.weight)
            else:
                cam = model.forward_cam(image)
            #print(cam1.size())


            # #修改部分
            # gradcam = GradCAM(model, 'fc8')
            # scores = model(image)
            # activation_maps = []  # 创建一个存储 CAM 的张量数组
            # for i in range(4):
            #     cam_result = gradcam(class_idx=i, scores=scores)
            #     #print(cam_result)
            #     cam_result = torch.stack(cam_result, dim=0) #列表内张量转张量
            #     #print(cam_result.size())
            #     activation_maps.append(cam_result)  # 将CAM结果添加到列表中

            # # 使用torch.cat将列表中的CAM结果拼接成一个张量
            # cam = torch.cat(activation_maps, dim=0).unsqueeze(0)
            # #修改部分

            if args.network_type == 'cls':
                cam = F.interpolate(cam, original_image_size, mode='bilinear', align_corners=False)[0]

                cam = cam.cpu().numpy() * label.reshape(args.num_classes, 1, 1)

                if i % 2 == 1:
                    cam = np.flip(cam, axis=-1)
                cam_list.append(cam)
            elif args.network_type == 'eps':
                cam = F.softmax(cam, dim=1)
                cam = F.interpolate(cam, original_image_size, mode='bilinear', align_corners=False)[0]
                
                cam_fg = cam[:-1]
                cam_bg = cam[-1:]

                cam_fg = cam_fg.cpu().numpy() * label.reshape(args.num_classes, 1, 1)
                cam_bg = cam_bg.cpu().numpy()

                if i % 2 == 1:
                    cam_fg = np.flip(cam_fg, axis=-1)
                    cam_bg = np.flip(cam_bg, axis=-1)
                cam_list.append((cam_fg, cam_bg))
            else:
                raise Exception('No appropriate model type')

    return cam_list


def _crf_with_alpha(image, cam_dict, alpha, t=10):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = imutils.crf_inference(image, bgcam_score, labels=bgcam_score.shape[0], t=t)
    n_crf_al = dict()
    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key+1] = crf_score[i+1]
    return n_crf_al

# 创建不同类别的颜色映射
def create_color_map(num_classes):
    # color_map = np.zeros((num_classes, 3), dtype=np.uint8)
    
    # # 为每个类别分配不同的颜色
    # for i in range(num_classes):
    #     # 这里示例为了简单，可以根据需要修改颜色分配
    #     color_map[i] = [i * 10, 255 - i * 10, i * 5]  # 这是一个示例，可以根据需要更改颜色

    # 为每个类别分配预定义的颜色
    if(num_classes==4):
        color_map = {
            0: [255, 255, 255],  # 背景
            1: [255, 0, 0],      # 类别1
            2: [0, 255, 0],      # 类别2
            3: [0, 0, 255],       # 类别3
            # 添加更多类别的颜色
        }
    else:
        color_map = {
            0: [0, 0, 0],  # 背景
            1: [255, 0, 0],       # 类别1
            2: [0, 255, 0],       # 类别2
            3: [0, 0, 255],       # 类别3
            4: [255, 255, 0],     # 类别4
            5: [0, 255, 255]      # 类别5
            # 添加更多类别的颜色
        }


    return color_map


def infer_cam_mp(image_ids, label_list, args):
    n_total_images = len(image_ids)
    #ReCAM
    if(args.recam):
        recam_predictor = Class_Predictor(4, 4096)
        recam_predictor = recam_predictor.cuda()
        recam_predictor.load_state_dict(torch.load(args.recam_weights))
        recam_predictor.eval()
    else:
        recam_predictor = None

    model = getattr(importlib.import_module(args.network), 'Net')(args.model_num_classes)
    model = model.cuda()
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    torch.no_grad()
    
    for i, (img_id, label) in enumerate(zip(image_ids, label_list)):
     
        # load image
        # 获取文件名前缀作为类别标识
        class_prefix = img_id.split("_")[0]
        class_prefix = int(class_prefix)
        if(class_prefix>0):#无缺陷图片不生成CAM
            img_path = os.path.join(args.img_root, img_id + '.jpg') # 构建图像文件路径
            img = Image.open(img_path).convert('RGB') # 打开图像并将其转换为RGB模式
            org_img = np.asarray(img) # 将图像转换为NumPy数组
            
            # infer cam_list
            cam_list = predict_cam(model, img, label, recam_predictor, args) # 使用模型生成CAM列表

            if args.network_type == 'cls':
                sum_cam = np.sum(cam_list, axis=0)  # 如果网络类型为'cls'，则将CAM列表求和
            elif args.network_type == 'eps':
                #cam_np = np.array(cam_list)
                cam_np = np.array(cam_list, dtype=object)

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
                
            

            #通用方案
            tensor[0, :, :] = args.thr  # 在第一个通道中添加阈值
            pred = np.argmax(tensor, axis=0).astype(np.uint8)  # 根据张量的最大值确定最终的预测

            # 创建分割掩码，将不同类别映射到不同的颜色
            pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
            for key in cam_dict.keys():
                pred_colored[pred==key+1] = color_map[key+1]  # 使用颜色映射为每个类别上色




            # save cam
            if args.cam_npy is not None:

                np.save(os.path.join(args.cam_npy, img_id + '.npy'), cam_dict) # 保存CAM字典为.npy文件

            if args.pred_cam is not None:
                imageio.imwrite(os.path.join(args.pred_cam, img_id + '.png'), pred_colored) # 保存预测结果为.png图像

            if args.crf is not None:
                for folder, t, alpha in args.crf_list:
                    cam_crf = _crf_with_alpha(org_img, cam_dict, alpha, t=t) # 使用CRF处理CAM
                    np.save(os.path.join(folder, img_id + '.npy'), cam_crf) # 保存处理后的CAM
            # 打印进度条
            pyutils.print_progress_bar(i + 1, n_total_images)
    print()


def main_mp(args):
    image_ids = load_img_id_list(args.infer_list)
    label_list = load_img_label_list_from_npy(image_ids, args.dataset)
    n_total_images = len(image_ids)
    print('n_total_images:', n_total_images)
    assert len(image_ids) == len(label_list)
    infer_cam_mp(image_ids, label_list, args)
    compute_miou(args.pred_cam,args.true_cam)
    



if __name__ == '__main__':
    crf_alpha = (4, 32)
    args = parse_args()
    main_mp(args)

    print(time.time() - start)
