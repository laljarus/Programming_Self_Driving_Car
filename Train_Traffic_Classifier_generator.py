# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 20:18:11 2018

@author: laljarus
"""

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten,Dropout,Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = 1+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

filename = "D:\\TrafficClassifier\\SimulatorData\\ImageLog_Train\\ImagesLabel.csv"
FolderPath = "D:\\TrafficClassifier\\SimulatorData\\ImageLog_Train\\"
lines = []

ImageFileNames = []
Labels = []

with open(filename) as csvfile:
    reader = csv.reader(csvfile)    
    for i,line in enumerate(reader):
        #if i <= len(reader):
        ImageFileNames.append(line[0].split("/")[-1])
        Labels.append(int(line[1]))
        
#plt.figure()
#plt.hist(Labels, bins='auto')  # arguments are passed to np.histogram
#plt.title("Histogram with 'auto' bins")
#plt.show()



classDict = {0:[1,0,0,0],1:[0,1,0,0],2:[0,0,1,0],4:[0,0,0,1]}

samples_dict = {'FileName':ImageFileNames,'Label':Labels}
samples_df = pd.DataFrame(samples_dict)


def Generator(samples,batch_size=32):
    num_samples = len(samples)
    
    while 1:
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]     
            
            arrImages = []
            arrLabels = []
            
            for ix,sample in batch_samples.iterrows():
                FileName = sample['FileName']
                Label    = sample['Label']
                
                img = cv2.imread(FolderPath+FileName)
                img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #img = cv2.resize(img,(160,320))
                arrImages.append(img)
                arrLabels.append(classDict[Label])
    
    
                if Label != 4:
                    img1 = cv2.flip(img,1)
                    arrImages.append(img1)
                    arrLabels.append(classDict[Label])
        
                    if Label !=0:
                        img2 = augment_brightness_camera_images(img)
                        arrImages.append(img2)
                        arrLabels.append(classDict[Label])
            
            X_train = np.array(arrImages)
            #X_normalized = X_train / 255.0 - 0.5
            y_train = np.array(arrLabels)
            yield shuffle(X_train, y_train)
    

#X_train = np.array(arrImages)
#y_train = np.array(arrLabels)
#y_one_hot = LabelBinarizer.fit_transform(y_train,4)


#plt.figure()
#plt.hist(arrLabels, bins='auto')  # arguments are passed to np.histogram
#plt.title("Histogram with 'auto' bins")
#plt.show()

train_samples,test_samples = train_test_split(samples_df)
# Generator to fetch the training data
train_generator = Generator(train_samples)
# Generator to fetch the test data
test_generator = Generator(test_samples)

model = Sequential()
#model.add(Cropping2D(cropping=((0,0), (0,0)),input_shape=(160,320,3)))
#model.add(Lambda(lambda x:x/255-0.5))
model.add(Convolution2D(6,(5, 5),activation = 'relu',input_shape=(600,800,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(16,(5, 5),activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(4))
model.add(Activation('relu'))


batch_size =32

adm = optimizers.adam(lr = 0.001)
model.compile(loss = "categorical_crossentropy",optimizer = adm,metrics=['accuracy'])
history = model.fit_generator(train_generator, len(train_samples)/batch_size, \
                              epochs =  5 , validation_data = test_generator,\
                              validation_steps= len(test_samples)/batch_size)

#history = model.fit(X_train,y_train,validation_split = 0.2,shuffle = True,epochs = 3,batch_size = 32)
					

model.save('model_train.h5')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model categorial crossentropy error loss')
plt.ylabel('crossentropy loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


"""
imgName = ImageFileNames[1600]
imgLabel = Labels[1600]
img = cv2.imread(FolderPath+imgName)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img1 = augment_brightness_camera_images(img)
img2 = cv2.flip(img,1)



plt.figure()
plt.imshow(img)

plt.figure()
plt.imshow(img1)
    
plt.figure()
plt.imshow(img2)
"""
