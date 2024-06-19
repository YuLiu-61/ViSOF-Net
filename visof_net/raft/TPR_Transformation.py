import numpy as np
import torchvision
import cv2 as cv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
import torch
import random

def TPR_Transformation(img_tensor,random):
    if(random==1):
        res = torch.roll(img_tensor, shifts=(-5, -5), dims=(1, 2))
        res[:,75:80,:] = 0
        res[:,:,75:80] = 0
    elif(random==2):
        res = torch.roll(img_tensor, shifts=(5, 5), dims=(1, 2))
        res[:, :5, :] = 0
        res[:, :, :5] = 0
    elif(random==3):
        res = torch.roll(img_tensor, shifts=(-5, 5), dims=(1, 2))
        res[:,75:80,:] = 0
        res[:, :, :5] = 0
    elif(random==4):
        res = torch.roll(img_tensor, shifts=(5, -5), dims=(1, 2))
        res[:, :5, :] = 0
        res[:, :, 75:80] = 0

    elif(random==5):
        distortion1 = int(torch.randint(0, 5, size=(1,)).item())
        distortion2 = int(torch.randint(0, 5, size=(1,)).item())
        width, height = F.get_image_size(img_tensor)
        topleft = [0+distortion1, 0+distortion2]
        topright = [width - 1-distortion1, 0+distortion2]
        botright = [width - 1, height - 1]
        botleft = [0, height - 1]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        res = F.perspective(img_tensor, startpoints, endpoints, InterpolationMode.BILINEAR, 0)
    elif(random==6):
        distortion1 = int(torch.randint(0, 5, size=(1,)).item())
        distortion2 = int(torch.randint(0, 5, size=(1,)).item())
        width, height = F.get_image_size(img_tensor)
        topleft = [0, 0]
        topright = [width-1, 0]
        botright = [width-1- distortion1, height-1- distortion2]
        botleft = [0+ distortion1, height - 1- distortion2]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        res = F.perspective(img_tensor, startpoints, endpoints, InterpolationMode.BILINEAR, 0)
    elif(random==7):
        distortion1 = int(torch.randint(0, 5, size=(1,)).item())
        distortion2 = int(torch.randint(0, 5, size=(1,)).item())
        width, height = F.get_image_size(img_tensor)
        topleft = [0+ distortion1, 0+ distortion2]
        topright = [width-1, 0]
        botright = [width-1, height-1]
        botleft = [0+ distortion1, height - 1- distortion2]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        res = F.perspective(img_tensor, startpoints, endpoints, InterpolationMode.BILINEAR, 0)
    elif(random==8):
        distortion1 = int(torch.randint(0, 5, size=(1,)).item())
        distortion2 = int(torch.randint(0, 5, size=(1,)).item())
        width, height = F.get_image_size(img_tensor)
        topleft = [0, 0]
        topright = [width-1- distortion1, 0+ distortion2]
        botright = [width-1- distortion1, height-1- distortion2]
        botleft = [0, height - 1]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        res=F.perspective(img_tensor, startpoints, endpoints, InterpolationMode.BILINEAR, 0)

    elif(random==9):
        transform = torchvision.transforms.RandomRotation([0,10])
        res = transform(img_tensor)
    elif(random==10):
        transform = torchvision.transforms.RandomRotation([-10,0])
        res=transform(img_tensor)
    return res