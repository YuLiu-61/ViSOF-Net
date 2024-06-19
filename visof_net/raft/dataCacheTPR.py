import cv2
try:
    import imageio
except ImportError:
    imageio = None
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import queue
from raft.TPR_Transformation import TPR_Transformation
import random

device = torch.device('cuda:0')


def dataCacheTPR(img):
    global rand
    rand=random.randint(1,10)
    res=TPR_Transformation(img,rand)
    return res

def dataCacheTrain5(img):
    if img.size(0) == 2:
        imgList = torch.split(img, [1, 1], dim=0)
        #print(img.size())
        resList = []
        for i in range(0, len(imgList)):
            res = torch.roll(imgList[i], shifts=(5, 5), dims=(2, 3))
            res[:, :, :5, :] = 0
            res[:, :, :, :5] = 0
            resList.append(res)

        img = torch.concat([resList[0], resList[1]], dim=0)
    elif img.size(0)==1:
        img = torch.roll(img, shifts=(5, 5), dims=(2, 3))
        img[:, :, :5, :] = 0
        img[:, :, :, :5] = 0
        img=img
    return img
def dataCacheTrain7(img):
    if img.size(0) == 2:
        #print(img.size())
        imgList = torch.split(img, [1, 1], dim=0)
        # print(img1.size())
        resList = []
        for i in range(0, len(imgList)):
            res = torch.roll(imgList[i], shifts=(-5, -5), dims=(2, 3))
            res[:, :, 635:640, :] = 0
            res[:, :, :, 635:640] = 0
            resList.append(res)

        img= torch.concat([resList[0], resList[1]], dim=0)
    elif img.size(0)==1:
        img = torch.roll(img, shifts=(-5, -5), dims=(2, 3))
        img[:, :, 635:640, :] = 0
        img[:, :, :, 635:640] = 0
        img=img
    return img


def dataCache():
    cache=queue.Queue(maxsize=43)
    return cache
#删除：cache.get()
#查看全部元素：cache.queue
#查看第一个元素：cache.queue[0]
#查看队列大小：cache.qsize()
cache=dataCache()



#17 frames for detect and test
def dataCacheTrainTPR(P3_t):
    if cache.full() == False:
        if (not cache.empty()) and (P3_t.size() != cache.queue[cache.qsize() - 1].size()):
            while not cache.empty():
                cache.get()
        cache.put(P3_t)
        P3_1 = dataCacheTPR(P3_t)
        P3_2 = P3_1
        P3_3 = P3_1
        P3_4 = P3_1
        P3_5 = P3_1
        P3_6 = P3_1
        P3_7 = P3_1
        P3_8 = P3_1
        P3_9 = P3_1
        P3_10 = dataCacheTPR(P3_t)
        P3_11 = P3_10
        P3_12 = P3_10
        P3_13 = P3_10
        P3_14 = P3_10
        P3_15 = P3_10
        P3_16 = P3_10
        P3_17 = P3_10

    else:
        if(P3_t.size()!=cache.queue[cache.qsize()-1].size()):
            while not cache.empty():
                cache.get()
            cache.put(P3_t)
            P3_1 = dataCacheTPR(P3_t)
            P3_2 = P3_1
            P3_3 = P3_1
            P3_4 = P3_1
            P3_5 = P3_1
            P3_6 = P3_1
            P3_7 = P3_1
            P3_8 = P3_1
            P3_9 = P3_1
            P3_10 = dataCacheTPR(P3_t)
            P3_11 = P3_10
            P3_12 = P3_10
            P3_13 = P3_10
            P3_14 = P3_10
            P3_15 = P3_10
            P3_16 = P3_10
            P3_17 = P3_10
        else:
            P3_1 = cache.queue[0]
            P3_2 = cache.queue[3]
            P3_3 = cache.queue[6]
            P3_4 = cache.queue[9]
            P3_5 = cache.queue[12]
            P3_6 = cache.queue[15]
            P3_7 = cache.queue[18]
            P3_8 = cache.queue[21]
            P3_9 = cache.queue[24]
            P3_10 = cache.queue[27]
            P3_11 = cache.queue[30]
            P3_12 = cache.queue[33]
            P3_13 = cache.queue[36]
            P3_14 = cache.queue[39]
            P3_15 = cache.queue[42]
            P3_16 = dataCacheTPR(P3_t)
            P3_17 = dataCacheTPR(P3_t)
            cache.get()
            cache.put(P3_t)
    return P3_1,P3_2,P3_3,P3_4,P3_5,P3_6,P3_7,P3_8,P3_9,P3_10,P3_11,P3_12,P3_13,P3_14,P3_15,P3_16,P3_17




