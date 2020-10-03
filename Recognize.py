# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:52:49 2020

@author: Ashutosh Mishra
"""


import os
import numpy as np
import urllib
import cv2
from keras.models import load_model
import pickle

with open('labels_mappings.p','rb') as f:
    labels = pickle.load(f)

model = load_model('face_recognition.h5')

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

URL = 'http://192.168.43.1:8080/shot.jpg'

def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    img = cv2.equalizeHist(img)
    img = cv2.resize(img,(200,200))
    img = img.reshape(1,200,200,1)
    img = img/255
    return img

ret = True

while ret:
    
    img = urllib.request.urlopen(URL)
    image = np.array(bytearray(img.read()),np.uint8)
    image = cv2.imdecode(image,-1)

    
    if image is not None:
        
        faces = classifier.detectMultiScale(image,1.2,5)
        
        if faces is not None:
            
            for x,y,w,h in faces:
                
                face_img = image[y:y+h,x:x+w].copy()
                
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
                
                pred = model.predict_classes(preprocess(face_img))
                
                cv2.putText(image,labels[pred[0]],(x,y),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
        
        cv2.imshow('video',image)
        
        if cv2.waitKey(1)==ord('q'):
            break
        
    else:
        break
    
cv2.destroyAllWindows()
