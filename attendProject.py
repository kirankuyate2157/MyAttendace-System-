import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime, date


path = 'adendee'
images = []
classNames = []
myList = os.listdir(path)
# print(myList)
for Tdata in myList:
    currentImg = cv2.imread(f'{path}/{Tdata}')
    images.append(currentImg)
    classNames.append(os.path.splitext(Tdata)[0])
