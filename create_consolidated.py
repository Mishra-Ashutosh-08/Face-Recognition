# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 16:56:02 2020

@author: Ashutosh Mishra
"""
import os
import cv2
import numpy as np
import pickle

im_dir = os.path.join(os.getcwd(),'images')
data_dir = os.path.join(os.getcwd(),'clean_data')


def preprocess(img):
    img = cv2.resize(img,(200,200))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img


data = []
labels = []

for i in os.listdir(im_dir):
    img = cv2.imread(os.path.join(im_dir,i))
    data.append(preprocess(img))
    labels.append(i.split('_')[0])

data = np.array(data)
labels = np.array(labels)

with open(os.path.join(data_dir,"images.p"),'wb') as f:
    pickle.dump(data,f)


with open(os.path.join(data_dir,"labels.p"),'wb') as f:
    pickle.dump(labels,f)





