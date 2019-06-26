import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from ann_model import ann_struct
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import normalize
def decode(data):
    decoded = []
    for i in range(data.shape[0]):
        decoded.append(np.argmax(data[i]))
    return np.array(decoded)
data_fid = open('data.pkl', 'rb')
[data, labels] = pickle.load(data_fid)
data_fid.close()
dataX = normalize(data,norm='max')
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
dataY = np_utils.to_categorical(encoded_Y)
(trainX, testX, trainY, testY) = train_test_split(dataX, dataY, test_size=0.10, random_state=42)
model = ann_struct()
model.load_weights('model-042-0.555556-0.900000.h5')
predictions = model.predict(testX)
result = decode(model.predict(testX))
print(result)
ref_result = decode(testY)
print(ref_result)
acc = 100 * (1 - float(np.count_nonzero(result - ref_result))/float(len(result)))
print('Acc = ' + str(acc))
