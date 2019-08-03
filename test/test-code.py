import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

raw_img = cv2.imread("../data/mofu.png")
resized_img = cv2.resize(raw_img, (224, 224))

plt.imshow(raw_img)
plt.show()
plt.imsave("./resized_img.png", resized_img)