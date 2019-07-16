import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize(model, x, layer_max_count):

    deconv_layers_list = []
    unpool_layers_list = []
    
    layer_count = 0

    for layer in model.features:

        if isinstance(layer, torch.nn.Conv2d):
            B, H, W, C = x.shape
            x = layer(x)
            deconv_layer = nn.ConvTranspose2d(layer.out_channels, H, layer.kernel_size, layer.stride, layer.padding)
            deconv_layer.weight = layer.weight
            deconv_layers_list.append(deconv_layer)

        if isinstance(layer, torch.nn.ReLU):
            x = layer(x)
            deconv_layers_list.append(layer)

        if isinstance(layer, torch.nn.MaxPool2d):
            x, index = layer(x)
            unpool_layers_list.append(index)
            unpool_layer = torch.nn.MaxUnpool2d(kernel_size=layer.kernel_size, stride=layer.stride,
                                               padding=layer.padding)
            deconv_layers_list.append(unpool_layer)

        layer_count += 1
        if layer_max_count == layer_count:
            break

    y = _max_feature(x[0]).unsqueeze_(0)

    _visualize(y, deconv_layers_list, unpool_layers_list)


def _max_feature(feature_maps):
    feature_maps_total_num = feature_maps.shape[0]

    activation_list = []
    for i in range(feature_maps_total_num):
        activation_val = torch.max(feature_maps[i, :, :])
        activation_list.append(activation_val.item())

    max_map_num = np.argmax(np.array(activation_list))
    max_map = feature_maps[max_map_num, :, :]
    max_activation_val = torch.max(max_map)

    max_map = torch.where(max_map == max_activation_val,
                          max_map,
                          torch.zeros(max_map.shape)
                          )

    for i in range(feature_maps_total_num):
        if i != max_map_num:
            feature_maps[i, :, :] = 0
        else:
            feature_maps[i, :, :] = max_map

    return feature_maps


def _visualize(y, deconv_layers_list, unpool_layers_list):
    for layer in reversed(deconv_layers_list):
        if isinstance(layer, nn.MaxUnpool2d):
            y = layer(y, unpool_layers_list.pop())
        else:
            y = layer(y)

    _imshow(y[0])


def _imshow(img):
    npimg = img.data.numpy()
    npimg = ((npimg - npimg.min()) * 255 / (npimg.max() - npimg.min())).astype(int)

    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()


if __name__ == '__main__':

    raw_img = cv2.imread("./data/cat.jpg")
    resized_img = cv2.resize(raw_img, (224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_img = transform(resized_img).unsqueeze_(0)

    model = models.vgg16(pretrained=True).eval()
    conv2d_layer_indices = []

    for i, layer in enumerate(model.features):
        if isinstance(layer, torch.nn.MaxPool2d):
            layer.return_indices = True
        #if isinstance(layer, torch.nn.Conv2d):
        if isinstance(layer, torch.nn.MaxPool2d):
            conv2d_layer_indices.append(i)

    for layer_max_count in conv2d_layer_indices:
        print("layer...%s" % layer_max_count)

        if layer_max_count == 30:
            visualize(model, input_img, layer_max_count)
