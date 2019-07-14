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
    print(type(features)) # <class 'torch.Tensor'>

    map_numbers = features.shape[0]

    act_lst = []
    for i in range(0, map_numbers):
        choose_map = features[i, :, :]
        activation = torch.max(choose_map)
        act_lst.append(activation.item())

    act_lst = np.array(act_lst)
    mark = np.argmax(act_lst)
    print(mark)

    choose_map = features[mark, :, :]

    max_activation = torch.max(choose_map)

    if mark == 0:
        features[1:, :, :] = 0
    else:
        print("zero1")
        features[:mark, :, :] = 0
        if mark != features.shape[1] - 1:
            print("zero2")
            features[mark + 1:, :, :] = 0

    choose_map = torch.where(choose_map == max_activation,
                             choose_map,
                             torch.zeros(choose_map.shape)
                             )

    features[mark, :, :] = choose_map

    print(int(max_activation))

    return features

def imshow(img):
    npimg = img.data.numpy()

    # 最小値を0,最大値を255に正規化
    npimg = ((npimg - npimg.min()) * 255 / (npimg.max() - npimg.min())).astype(int)

    # 要素の順番を(RGB, H, W) から (H, W, RGB)に変更
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()

# Read image
img = cv2.imread("./data/plane.jpg")#, cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = transform(img)
img.unsqueeze_(0)

x = img
deconv_list = []
unpool_list = []

print("start loop")

num = 0

model = models.vgg16(pretrained=True).eval()

for layer in model.features:
    if isinstance(layer, torch.nn.MaxPool2d):
        layer.return_indices = True

for layer in model.features:

    if isinstance(layer, torch.nn.Conv2d):
        print("Conv2d...%s" % layer)
        B, H, W, C = x.shape
        x = layer(x)
        deconvLayer = nn.ConvTranspose2d(layer.out_channels, H, layer.kernel_size, layer.stride, layer.padding)

        print("deConv2d...%s" % deconvLayer)
        deconvLayer.weight = layer.weight
        deconv_list.append(deconvLayer)

    if isinstance(layer, torch.nn.ReLU):
        x = layer(x)
        deconv_list.append(layer)

    if isinstance(layer, torch.nn.MaxPool2d):
        print("MaxPool2d...%s" % layer)
        x, index = layer(x)
        unpool_list.append(index)
        unpoolLayer = torch.nn.MaxUnpool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)#, dilation=1, ceil_mode=False))
        print("MaxUnpool2d...%s" % unpoolLayer)
        deconv_list.append(unpoolLayer)

    #num += 1
    #if num == 28:
    #    break

print("loop end")

x = max_feature(x[0])
y = x.unsqueeze_(0)
# y = x

result = y.clone()


for layer in reversed(deconv_list):
    if isinstance(layer, nn.MaxUnpool2d):
        y = layer(y, unpool_list.pop())
        print("unpool_list length: %s" % len(unpool_list))
    else:
        y = layer(y)

imshow(y[0])

plt.savefig('result.jpg')


