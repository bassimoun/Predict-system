import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.initializers import glorot_normal
from cnn_struct import created_model

data_fid = open('data.pkl', 'rb')
[dataX, labels] = pickle.load(data_fid)
data_fid.close()
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
dataY = np_utils.to_categorical(encoded_Y)
(trainX, testX, trainY, testY) = train_test_split(dataX, dataY, test_size=0.10, random_state=42)
model = created_model()
model.load_weights('models\\model-001-0.888818-0.833333.h5')
# print("Loaded model from disk")
predictions = model.predict(testX)
#print(predictions)
score = np.array([np.argmax(x) for x in predictions])
print(score)
ref = np.array([np.argmax(x) for x in testY])
print(ref)
acc=100*(1-float(np.count_nonzero(ref-score))/float(len(ref)))
print(acc)