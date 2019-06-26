import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
import glob
import random
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Activation, Conv2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, CSVLogger
from keras.initializers import glorot_normal
import sys
data = []
img = cv2.imread(sys.argv[1])
image = cv2.resize(img, (224, 224)) ## resize image to 224*224
data.append(image.reshape(224, 224, 3))
dataX = np.array(data, dtype="float") / 255.0      # Normalize all pixels


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('model-001-0.888818-0.833333.h5')
# print("Loaded model from disk")

predictions = model.predict(dataX)
# print(predictions)
fid = open('ces_normal_prob.txt','w')
fid.write(str(100*predictions[0][0])+' %\n')
fid.write(str(100*predictions[0][1])+' %\n')
fid.close()