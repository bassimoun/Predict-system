from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from numpy.random import seed
from ann_model import ann_struct
from sklearn.preprocessing import LabelEncoder
seed(42)
np_epoch = 50
data_fid = open('data.pkl', 'rb') 
[data, labels] = pickle.load(data_fid)
data_fid.close()
X = normalize(data,norm='max')
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
dataY = np_utils.to_categorical(encoded_Y)
X_train, X_test, y_train, y_test = train_test_split(X, dataY, test_size=0.1, random_state=42)
learning_rate = 0.01
model = ann_struct()
sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
checkpoint = ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='auto')
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=np_epoch, callbacks=[checkpoint])
np.savetxt('my_dnn_model_loss.csv', hist.history['loss'])
np.savetxt('my_dnn_model_val_loss.csv', hist.history['val_loss'])


