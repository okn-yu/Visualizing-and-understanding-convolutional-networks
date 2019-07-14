import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt

def max_feature(features):

    # 全512枚の特徴マップから活性値が最大値を取得する
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
    npimg = ((npimg - npimg.min()) * 255 / (npimg.max() - npimg.min())).astype(int)

    # 要素の順番を(RGB, H, W) から (H, W, RGB)に変更
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()

def visualize(model, x, num):
    print("loop start")

    count = 0

    deconv_list = []
    unpool_list = []
    features_list = []

    for layer in model.features:

        if isinstance(layer, torch.nn.Conv2d):
            # print("Conv2d...%s" % layer)
            B, H, W, C = x.shape
            x = layer(x)
            deconvLayer = nn.ConvTranspose2d(layer.out_channels, H, layer.kernel_size, layer.stride, layer.padding)

            # print("deConv2d...%s" % deconvLayer)
            deconvLayer.weight = layer.weight
            deconv_list.append(deconvLayer)

        if isinstance(layer, torch.nn.ReLU):
            x = layer(x)
            deconv_list.append(layer)

        if isinstance(layer, torch.nn.MaxPool2d):
            # print("MaxPool2d...%s" % layer)
            x, index = layer(x)
            unpool_list.append(index)
            unpoolLayer = torch.nn.MaxUnpool2d(kernel_size=layer.kernel_size, stride=layer.stride,
                                               padding=layer.padding)  # , dilation=1, ceil_mode=False))
            # print("MaxUnpool2d...%s" % unpoolLayer)
            deconv_list.append(unpoolLayer)

        features_list.append(x.clone())

        count += 1
        if count == num:
            break

    print("loop end")

    x = max_feature(x[0])
    y = x.unsqueeze_(0)

    _visualize(deconv_list, y, unpool_list)

def _visualize(deconv_list, y, unpool_list):
    for layer in reversed(deconv_list):
        if isinstance(layer, nn.MaxUnpool2d):
            y = layer(y, unpool_list.pop())
        else:
            y = layer(y)

    imshow(y[0])


# Read image
img = cv2.imread("./data/cat.jpg")#, cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))

transform = transforms.Compose([
    transforms.ToTensor()#,
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = transform(img)
img.unsqueeze_(0)

x = img
model = models.vgg19(pretrained=True).eval()

conv2d_list = []

# -> リスト内包で実装
for i, layer in enumerate(model.features):
    if isinstance(layer, torch.nn.MaxPool2d):
        layer.return_indices = True
    if isinstance(layer, torch.nn.Conv2d):
        conv2d_list.append(i)

print(conv2d_list)

for i in conv2d_list:
    visualize(model, x, i)




