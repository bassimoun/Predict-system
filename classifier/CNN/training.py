import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.initializers import glorot_normal
from cnn_struct import created_model
seed = 42
init = glorot_normal()
epoch = 200
BS = 32
num_classes = 2
random.seed(seed)
INIT_LR = 0.01
data_fid = open('data.pkl', 'rb')
[dataX, labels] = pickle.load(data_fid)
data_fid.close()
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
dataY = np_utils.to_categorical(encoded_Y)
(trainX, testX, trainY, testY) = train_test_split(dataX, dataY, test_size=0.10, random_state=42)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
model = created_model()
checkpoint = ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='auto')
csv_logger = CSVLogger('report/log_' + str(INIT_LR) + '.csv', append=False, separator=';')
sgd = optimizers.SGD(lr=INIT_LR, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX), epochs=epoch, callbacks=[csv_logger, checkpoint])