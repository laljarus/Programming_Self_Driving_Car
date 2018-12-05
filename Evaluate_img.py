# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 23:41:30 2018

@author: laljarus
"""

#import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
import h5py
from keras import __version__ as keras_version
import time

filename = "D:\\TrafficClassifier\\SimulatorData\\ImageLog_Test\\ImagesLabel_easy.csv"
FolderPath = "D:\\TrafficClassifier\\SimulatorData\\ImageLog_Test\\"
lines = []

#classDict = {[1,0,0,0]:0,[0,1,0,0]:1,[0,0,0,1]:2,[0,0,0,1]:4}
def normalize_min_max(x_data,x_min = 0,x_max = 255,a = 0,b =1):
    return a + (x_data-x_min)*(b-a)/(x_max-x_min)

img_id = 200

imgpath = FolderPath+"image_"+str(img_id)+".png"
img_width, img_height = 600, 800

img = cv2.imread(imgpath)
img = np.expand_dims(img, axis=0)

images = np.vstack([img])

img = image.load_img(imgpath, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images1 = np.vstack([x])

image = Image.open(imgpath)
image_array = np.asarray(image)
image_array = image_array[None,:,:,:]

#img_norm = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#img_resize = np.reshape(img_norm,[1,600,800,3])

f = h5py.File('model_test.h5', mode='r')
model_version = f.attrs.get('keras_version')
keras_version = str(keras_version).encode('utf8')

if model_version != keras_version:
    print('You are using Keras version ', keras_version,
          ', but the model was built using ', model_version)

model = load_model('model_test.h5')
t1 = time.time()
class_arr = model.predict(image_array, batch_size=1)
t2 = time.time()

print(class_arr)
print(t2-t1)
img = cv2.imread(imgpath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img)
 
 