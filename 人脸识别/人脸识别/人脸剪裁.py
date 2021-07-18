# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:47:51 2020

@author: yjw
"""
import os
import torchvision.models as models
from PIL import Image, ImageDraw
import datetime
from torch.autograd import Variable
import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN, extract_face
import matplotlib.pyplot as plt
from torch import nn, optim
import sys
def processing():#图片的预处理，去除背景并且使得图片的格式满足模型输入的格式
    lista=os.listdir("test")
    lista.sort()
    for item in lista:
        test_path = "test/" + item
        print(item)
        mtcnn = MTCNN(keep_all=True)
        img = Image.open(test_path)
        boxes, probs, points = mtcnn.detect(img, landmarks=True)
        for i, (box, point) in enumerate(zip(boxes, points)):
            img.crop(box).save(item)
processing()