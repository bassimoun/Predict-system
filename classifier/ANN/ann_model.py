# coding:utf-8
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout

def ann_struct():
    model = Sequential()
    model.add(Dense(16, input_shape=(12,)))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model_json = model.to_json()
    with open('model.json', "w") as json_file:
        json_file.write(model_json)

    return model


