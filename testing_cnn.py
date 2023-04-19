# tensor_folder = tf.summary.create_file_writer("tensor_board/test_images/")

from vgg import define_model
from img_data_generator import img_data_generator
import time
import pandas as pd

import os



import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"







def run_test():
    #models = [[1,"VGG1"],[3, "VGG3"],[3, "Data Augmentation"], [16, "VGG16"]]
    models = [[16, "VGG16"]] 
    training_time = []
    training_loss=[]
    training_acc=[]
    testing_acc=[]
    param =[]
    for [i, j] in models:
        # tf_train = tf.summary.create_file_writer("tensor_board")
        # test_writer = tf.summary.create_file_writer(logdir_test)
        model = define_model(i)
        print("*"*50, i, ",", j, "*"*50)
        train_it, test_it  = img_data_generator(j)
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq='batch')
        start = time.time()
        history = model.fit(train_it, steps_per_epoch=len(train_it),validation_data=test_it, validation_steps=len(test_it), epochs=1, verbose=1)
        end = time.time()

        time_taken = end -start 
        _, acc_test = model.evaluate(test_it, steps=len(test_it), verbose=0)
        print("Testing accuracy: ", '> %.3f' % (acc_test * 100.0))
        _, acc_train = model.evaluate(train_it, steps=len(train_it), verbose=0)
        print("Training accuracy: ",'> %.3f' % (acc_train * 100.0))
        print(model.summary())
        no_of_params=model.count_params()
        print("Number of parameters: ",no_of_params)
        loss = history.history['loss'][-1]
        print('Training loss: ', history.history['loss'][-1])
        training_time.append(time_taken)
        training_loss.append(loss)
        training_acc.append(acc_train)
        testing_acc.append(acc_test)
        param.append(no_of_params)
        # summary_writer = tf.summary.create_file_writer(log_dir)
        # with summary_writer.as_default():
        #  for epoch in range(1):
        #          print(history.history)
        #          tf.summary.scalar('Accuracy', model.history.history['accuracy'][epoch], step=epoch)
        #          print(history.history['accuracy'][epoch])
        # print("hi")
        # summary_writer.close()

    df = pd.DataFrame({"Training time": training_time, "Training loss": training_loss, "Training accuracy": training_acc, "Testing accuracy": testing_acc, "Number of model parameters": param}, index=["VGG1", "VGG3", "VGG3 with data aug", "Transfer learning-VGG16"])
    df = df.reset_index(drop=True)
    print(df)
    return df
print(run_test())
