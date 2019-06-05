#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function


# In[2]:


import numpy as np
import scipy as sp
import pandas as pd
from numpy.random import rand

#sklearn imports for metrics
from sklearn import preprocessing
from sklearn.metrics import auc, precision_recall_curve, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

#Matplotlib imports for graphs
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import tensorflow as tf
import keras

# Models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

# Layers
from keras.layers import Dense, Activation, Flatten, Dropout, GlobalMaxPooling2D,BatchNormalization
from keras import backend as K

# Other
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam, Adagrad
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model


# In[4]:


train_data_dir = 'D:\\catvsdog\\training_set'
validation_data_dir = 'D:\\catvsdog\\test_set'
nb_train_samples = 200
nb_validation_samples = 80
epochs = 40
batch_size = 16
img_width, img_height = 224, 224


# In[6]:


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()


# In[7]:


model = Sequential()

for layer in vgg16_model.layers[:-1]:
    model.add(layer)
model.summary()


# In[8]:


for layer in model.layers:
    layer.trainable = False


# In[9]:


#top_model = Sequential()
#top_model.add(Flatten(input_shape=model.output_shape[1:]))
#top_model.add(Dense(256, activation='relu'))
#top_model.add(Dropout(0.5))
#top_model.add(Dense(1, activation='sigmoid'))
#model.summary()

#model.add(GlobalMaxPooling2D())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "sigmoid")) #as there are 9 classes
model.summary()


# In[10]:


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# In[13]:


train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale =1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'binary')


# In[14]:


model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)


# In[ ]:




