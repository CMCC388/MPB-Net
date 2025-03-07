import cv2
import numpy as np
from torchcam.methods import GradCAM
import torch
import importlib
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize
import torch.nn.functional as F
network = 'network.resnet38_eps'
model_num_classes = 4
img = read_image("/home/caojun/data/SD-saliency-900/JPEGImages/3_2.jpg")
img = img.cuda()
original_image_size = np.asarray(img.cpu()).shape[-2:]
print(original_image_size)
img = img.unsqueeze(0)
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


print(img.shape)


model = getattr(importlib.import_module(network), 'Net')(model_num_classes)
model = model.cuda()
model.load_state_dict(torch.load('xiaorong/checkpoint_eps_112_0.7256.pth'))
model.eval()
#torch.no_grad()

scores = model.forward(input_tensor)
print('输出scores:')
print(scores.size())
cam_extractor = GradCAM(model,target_layer='fc8')
class_idx = scores.squeeze(0).argmax().item()
print('输出class_idx:')
print(class_idx)
gradcam = cam_extractor(class_idx=class_idx,scores=scores)
# # 前向传播获取特征图


print(gradcam.shape)
cam = gradcam


# 计算CAM
cam = cam[:,2, :, :]
cam = cam.squeeze()  # 将张量压缩为二维
print(cam.shape)
#cam = F.relu(cam)  # ReLU激活
cam = cam.detach().cpu().numpy()

# 归一化CAM
cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # 归一化到 0~1 之间
# 将CAM调整为与输入图像相同大小
print(cam.shape)
cam = cv2.resize(cam, (200, 200))
# 将CAM转换为热力图并与原始图像叠加
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
result = cv2.addWeighted(np.uint8(255 * img.cpu().numpy().squeeze().transpose(1, 2, 0)), 0.5, heatmap, 0.5, 0)



# 保存结果
cv2.imwrite('hotmap_eps/3.2_g.jpg', result)