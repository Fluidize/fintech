#filter - methods to extract specific features
#kernel - applied to an image to extract certain features
#stride - pixels intervals the kernel skips
#padding - zeroes on the edges of image

# import os

# import gradio as gr
import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import plotly.express as px

import skimage
import torch
import torch.nn.functional as F
# import torch.nn as nn
# import torchvision.datasets as datasets
# import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
# from torch.utils.data import DataLoader
# from tqdm.notebook import tqdm
from PIL import Image

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_mnist = datasets.MNIST(
#     root='dataset/',
#     train=True,
#     transform=Compose([RandomCrop(24), ToTensor]),
#     download=True
# )
# test_mnist = datasets.MNIST(
#     root='dataset/',
#     train=False,
#     transform=Compose([RandomCrop(24), ToTensor]),
#     download=True
# )

# train_dataset = datasets.KMNIST(
#     root='dataset/',
#     train=True,
#     transform=Compose([RandomCrop(24), ToTensor()]),
#     download=True
# )
# test_dataset = datasets.KMNIST(
#     root='dataset/',
#     train=False,
#     transform=Compose([RandomCrop(24), ToTensor()]),
#     download=True
# )

cameraman = Image.fromarray(skimage.data.camera())
transform = Compose([
    ToTensor(),
    Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
])
cameraman = transform(cameraman)

horiz_filter = torch.tensor([[[
    [1.0,2.0,1.0],
    [0.0,0.0,0.0],
    [-1.0,-2.0,-1.0]
    ]]])
vert_filter = torch.tensor([[[
    [1.0,0.0,-1.0],
    [2.0,0.0,-2.0],
    [1.0,0.0,-1.0]
    ]]])

average_filter = torch.ones((1, 1, 7, 7)) / 49.

filtered = F.conv2d(cameraman, average_filter)

plt.imshow(filtered[0], cmap='Greys_r')
plt.show()