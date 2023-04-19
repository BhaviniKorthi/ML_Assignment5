import os 
import re
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.image import imread
from shutil import copyfile
from random import seed
from random import random

import numpy as np
import cv2

# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array


print(os.listdir('images\BEAR'))
print(os.listdir('images\SHEEP'))


#code to keep only 100 images in each folder
filenames = os.listdir('images/SHEEP')

for filename in filenames:
    # Use regex to extract the number from the filename
    match = re.search(r'\d+', filename)
    if match:
        number = int(match.group())
    
        if number >= 100:
            os.remove('images/SHEEP//'+filename)
            
print("no of images" , len(os.listdir('images/SHEEP')))
print("no of images",len(os.listdir('images/BEAR')))



# code to display images in the dataset
def display_images(folder):

    f1 = os.listdir(folder)[:9]
    print(f1)
    j = 0   
    for i in f1:
        pyplot.subplot(330 + 1 + j)
        image = imread(folder+'/'+i)
        pyplot.imshow(image)
        j += 1  
    plt.show()
# display_images('images/BEAR')
# display_images("images/SHEEP")



def reshape_images(folder,class_name,folder_name):
    f1 = os.listdir(folder)
    images, labels = list(), list()
    for i in f1:
        
        image = cv2.imread(folder+"/"+i)
        image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_LINEAR)
        images.append(image)
        labels.append(class_name)
    images = np.array(images)
    np.save(folder_name + '.npy', images)
  
try:
    os.mkdir("np_images/BEAR")   
except:
    print("folder already exists")
# os.makedirs("np_images/SHEEP")
# os.makedirs("np_images/BEAR")
# reshape_images("images/BEAR",0,"np_images/BEAR/bear")
# reshape_images("images/SHEEP",1,"np_images/SHEEP/sheep")


# def split_data(folder_name,target_floder):
    
#     images = np.load(folder_name)
#     print(images.shape)
#     for i in range(images.shape[0]):
#         image = images[i]
#         cv2.imwrite(target_floder+"/"+str(i)+".jpg",image)


# photos = np.load('np_images/BEAR/bear.npy')
# print(photos.shape)
# print(photos)

