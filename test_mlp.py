

from vgg import define_model
from img_data_generator import img_data_generator
import time
import pandas as pd
from mlp import mlp
def run_test():
    # models = [[1, "VGG1"]] 
    training_time = []
    training_loss=[]
    training_acc=[]
    testing_acc=[]
    param =[]
    model  = mlp()


    train_it, test_it  = img_data_generator("VGG1")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq='batch')
    start = time.time()
    history = model.fit(train_it, steps_per_epoch=len(train_it),validation_data=test_it, validation_steps=len(test_it), epochs=5, verbose=0)
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

    df = pd.DataFrame({"Training time": training_time, "Training loss": training_loss, "Training accuracy": training_acc, "Testing accuracy": testing_acc, "Number of model parameters": param}, index=["VGG1", "VGG3", "VGG3 with data aug", "Transfer learning-VGG16"])
    df = df.reset_index(drop=True)
    return df
print(run_test())
