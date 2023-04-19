from img_data_generator import img_data_generator
from vgg import define_model
import os
import cv2

    
import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard


from keras.preprocessing.image import ImageDataGenerator

def run_test():
    models = [[1,"VGG1"],[3, "VGG3"],[3, "Data Augmentation"], [16, "VGG16"]]
    
    training_time = []
    training_loss=[]
    training_acc=[]
    testing_acc=[]
    step = 0
    param =[]
    
    for [i, j] in models:
        model = define_model(i)
        print("*"*50, i, ",", j, "*"*50)
        train_it, test_it  = img_data_generator(j)

        train_writer = tf.summary.create_file_writer("logs/" + j  + "/train/")
        test_writer = tf.summary.create_file_writer("logs/" + j  + "/test/")
        train_step = test_step = 0
        for epoch in range(5):
            train_it.reset
            for batch in range(len(train_it)):
                x_train, y_train = train_it.next()
                train_loss, train_acc = model.train_on_batch(x_train, y_train)

                with train_writer.as_default():
                    tf.summary.scalar("TRAIN Loss", train_loss, step=train_step)
                    tf.summary.scalar("TRAIN Accuracy", train_acc, step=train_step)
                    train_step += 1

                val_loss, val_acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
                with test_writer.as_default():
                    tf.summary.scalar(" TEST Loss", val_loss, step=test_step)
                    tf.summary.scalar("TEST Accuracy", val_acc, step=test_step)
                    test_step += 1
        test_it.reset()
        if j == "VGG16":
            vgg = model
        print(model.predict(test_it))
        prediction = model.predict(test_it)
        sum_ = 0
        for i in range(len(prediction)):
            prediction[i] = round(prediction[i][0]) 
            sum_ += prediction[i][0]
        print("sum",sum_)
        print(prediction)
        test_it.reset()
        _, acc_test = model.evaluate(test_it, steps=len(test_it), verbose=0)
        print("Testing accuracy: ", '> %.3f' % (acc_test * 100.0))
        
        labels = ["BEAR", "SHEEP"]


        

        test_it.reset()
        images,y=test_it.next()
        predict = []
        if j != "VGG16":
            for k in range(40):
                image = images[k]
                if  j != "VGG16": 
                   image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_LINEAR)
                   image = image.reshape(1,200,200,3)
                else:
                    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
                    image = image.reshape(1,224,224,3)
                predict.append(np.where(model.predict(image)[0]>=0.5, 1, 0))

        if j == "VGG16":
            images = []
            for k in os.listdir("bear_vs_sheep/test/"):
                for i in os.listdir("bear_vs_sheep/test/" + k):
                    image = cv2.imread("bear_vs_sheep/test/"+k+'/'+i)
                    images.append(image)
                    image_array = np.array(image,dtype=np.float32)
                    mean = [123.68, 116.779, 103.939]
                    image_array -= mean
                    image = cv2.resize(image_array, (224, 224), interpolation=cv2.INTER_LINEAR)
                    image = image.reshape(1,224,224,3)
                    predict.append(np.where(model.predict(image)[0]>=0.5, 1, 0))

                
        predict = np.array(predict).flatten()
        print (np.array(y),predict,prediction)
        figure = image_grid(images, predict, labels)
        file_writer = tf.summary.create_file_writer("logs/" + j  + "/test_images/")
        with file_writer.as_default():
            tf.summary.image("predicted Images", plot_to_image(figure), step=step)
            step += 1

    return


import io
import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

def image_grid(images, labels, class_names, rows=5, cols=8):
    """
    Create a grid of images with corresponding labels.
    """
    # Create the figure and axis objects.
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

    # Iterate over the images and labels and plot them in the grid.
    for i, ax in enumerate(axs.flatten()):
        # Plot the image.
        ax.imshow(images[i], cmap='gray')

        # Set the label.
        class_idx = labels[i]
        print(class_idx)
        class_name = class_names[class_idx]
        print(class_name)
        ax.set_title(class_name)

        # Remove the axis ticks and labels.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Adjust the spacing between the subplots.
    plt.tight_layout()

    return fig


def plot_to_image(figure):
    """
    Convert a Matplotlib figure to a PNG image and return it as a TensorFlow tensor.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    # Decode the PNG image into a TensorFlow tensor.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension.
    image = tf.expand_dims(image, 0)

    return image


        
        
        
        

run_test()



