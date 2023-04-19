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


def img_data_generator(type):
    # tensorboard_callback = TensorBoard(log_dir='./logs/{}'.format(type))
    if type == "VGG1" or type =="VGG3":
        datagen = ImageDataGenerator(rescale=1.0/255.0)
        train_it = datagen.flow_from_directory("bear_vs_sheep/train",
        class_mode='binary', batch_size=64, target_size=(200, 200))
        test_it = datagen.flow_from_directory("bear_vs_sheep/test",
        class_mode='binary', batch_size=64, target_size=(200, 200))


    elif type == "VGG16":
        datagen = ImageDataGenerator(featurewise_center=True)
        datagen.mean = [123.68, 116.779, 103.939]
        train_it = datagen.flow_from_directory("bear_vs_sheep/train",
        class_mode='binary', batch_size = 64 , target_size=(224, 224))
        test_it = datagen.flow_from_directory("bear_vs_sheep/test",
        class_mode='binary', batch_size= 40, target_size=(224, 224))


    elif type == "Data Augmentation":
        train_datagen = ImageDataGenerator(rescale=1.0/255.0,
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        train_it = train_datagen.flow_from_directory("bear_vs_sheep/train",
        class_mode='binary', batch_size=64, target_size=(200, 200))
        test_it = test_datagen.flow_from_directory("bear_vs_sheep/test",
        class_mode='binary', batch_size=40, target_size=(200, 200))


    return train_it, test_it