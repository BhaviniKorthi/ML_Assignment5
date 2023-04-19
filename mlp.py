from keras.models import Sequential
from keras.layers import Dense, Flatten

def mlp():

    model = Sequential()
    model.add(Flatten(input_shape=(200, 200, 3)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
