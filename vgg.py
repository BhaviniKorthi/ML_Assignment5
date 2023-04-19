import os
import time
import tensorboard
import tensorflow as tf
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  


import warnings
warnings.filterwarnings("ignore")

from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from img_data_generator import img_data_generator

def define_model(blocks):
    if (blocks == 1):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    elif (blocks == 3):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy']) 
        return model
    
    if (blocks == 16):
        model = VGG16(include_top=False, input_shape=(224, 224, 3))
        for layer in model.layers:
            layer.trainable = False
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = Dense(1, activation='sigmoid')(class1)
        model = Model(inputs=model.inputs, outputs=output)
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model






