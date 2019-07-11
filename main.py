import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("./data/Moomin.jpg", cv2.IMREAD_COLOR)
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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = transform(img)
img.unsqueeze_(0)

print(img.shape)
print(type(img))

for layer in model.features:
    print(layer)
    img = layer(img)
    print(img.shape)
    # return_indice = Trueだとtupleが返る

# model(img)