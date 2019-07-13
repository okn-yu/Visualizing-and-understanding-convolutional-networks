import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt

def max_feature(features):

    # 全512枚の特徴マップから活性値が最大値を取得する
    print("max_feature")
    print(features.shape)

def imshow(img):
    npimg = img.data.numpy()

    # 最小値を0,最大値を255に正規化
    npimg = ((npimg - npimg.min()) * 255 / (npimg.max() - npimg.min())).astype(int)

    # 要素の順番を(RGB, H, W) から (H, W, RGB)に変更
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()

# Read image
img = cv2.imread("./data/cat.jpg", cv2.IMREAD_COLOR)
# img = img[:, :, [2, 1, 0]]


# H, W, C = img.shape
# print(img.shape)
img = cv2.resize(img, (224, 224))
# print(img.shape)

# cv2.imwrite("out.jpg", img)
# cv2.imshow("result", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

model = models.vgg16(pretrained=True)

for layer in model.features:
    if isinstance(layer, torch.nn.MaxPool2d):
        layer.return_indices = True

transform = transforms.Compose([
    transforms.ToTensor()#,
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = transform(img)
img.unsqueeze_(0)

# print(img.shape)
# print(type(img))

x = img
convd_list = []
deconv_list = []
unpool_list = []

print("start loop")

num = 0

for layer in model.features:

    if isinstance(layer, torch.nn.Conv2d):
        print("isinstance...%s" % layer)
        B, H, W, C = x.shape
        print(B, H, W, C)
        x = layer(x)
        deconvLayer = nn.ConvTranspose2d(layer.out_channels, H, layer.kernel_size, layer.stride, layer.padding)
        deconv_list.append(deconvLayer)

    if isinstance(layer, torch.nn.ReLU):
        deconv_list.append(layer)

    if isinstance(layer, torch.nn.MaxPool2d):
        print("isinstance...%s" % layer)
        x, index = layer(x)
        print(x.shape)
        unpool_list.append(index)
        deconv_list.append(torch.nn.MaxUnpool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding))#, dilation=1, ceil_mode=False))
        num += 1
        if num == 5:
            break

print("loop end")

print(x.shape) # -> torch.Size([1, 512, 7, 7])

max_feature(x[0])

y = x

#y = unpool_list[0](x, index)
#print(y.shape)

for layer in reversed(deconv_list):

    if isinstance(layer, nn.MaxUnpool2d):
        # print(layer)
        # print(y.shape)
        y = layer(y, unpool_list.pop())
    else:
        # print(layer)
        # print(y.shape)
        y = layer(y)

imshow(y[0])

