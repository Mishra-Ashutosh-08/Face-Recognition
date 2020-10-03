# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 07:42:41 2020

@author: Ashutosh Mishra
"""
import os
import numpy as np
import urllib
import cv2

classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

im_dir = os.path.join(os.getcwd(),'images')

URL = 'http://192.168.43.1:8080/shot.jpg'

ret = True

data = []

while ret:
    
    img = urllib.request.urlopen(URL)
    image = np.array(bytearray(img.read()),np.uint8)
    image = cv2.imdecode(image,-1)
    image = cv2.resize(image,(640,480))
    
    if image is not None:
        
        faces = classifier.detectMultiScale(image,1.2,5)
        
        if faces is not None:
            
            for x,y,w,h in faces:
                
                face_img = image[y:y+h,x:x+w].copy()
                
                if len(data)<200:
                    data.append(face_img)
                
                else:
                    cv2.putText(image,'complete',(200,200),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),5)
                
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),5)
        
        cv2.imshow('video',image)
        
        if cv2.waitKey(1)==ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()

name = input('enter any name: ')
c=0
for i in data:
    cv2.imwrite(os.path.join(im_dir,name+'_'+str(c)+'.jpg'),i)
    c+=1