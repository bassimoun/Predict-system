from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Activation, Conv2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.initializers import glorot_normal
seed = 42
init = glorot_normal()
def created_model():
    images = Input(shape=(224, 224, 3))
    Conv_1 = Conv2D(32, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(images)
    BN_Conv_1 = BatchNormalization()(Conv_1)
    Act_Conv_1 = Activation('relu')(BN_Conv_1)
    Pool_Conv_1 = MaxPooling2D(pool_size=(2, 2))(Act_Conv_1)
    Conv_2 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(Pool_Conv_1)
    BN_Conv_2 = BatchNormalization()(Conv_2)
    Act_Conv_2 = Activation('relu')(BN_Conv_2)
    Pool_Conv_2 = MaxPooling2D(pool_size=(2, 2))(Act_Conv_2)
    Conv_3 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(Pool_Conv_2)
    BN_Conv_3 = BatchNormalization()(Conv_3)
    Act_Conv_3 = Activation('relu')(BN_Conv_3)
    Pool_Conv_3 = MaxPooling2D(pool_size=(2, 2))(Act_Conv_3)
    Conv_4 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(Pool_Conv_3)
    BN_Conv_4 = BatchNormalization()(Conv_4)
    Act_Conv_4 = Activation('relu')(BN_Conv_4)
    Pool_Conv_4 = MaxPooling2D(pool_size=(2, 2))(Act_Conv_4)
    Conv_5 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(Pool_Conv_4)
    BN_Conv_5 = BatchNormalization()(Conv_5)
    Act_Conv_5 = Activation('relu')(BN_Conv_5)
    Pool_Conv_5 = MaxPooling2D(pool_size=(2, 2))(Act_Conv_5)
    Conv_6 = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(Pool_Conv_5)
    BN_Conv_6 = BatchNormalization()(Conv_6)
    Act_Conv_6 = Activation('relu')(BN_Conv_6)
    Pool_Conv_6 = MaxPooling2D(pool_size=(2, 2))(Act_Conv_6)
    Conv_7 = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer=init, padding='same')(Pool_Conv_6)
    BN_Conv_7 = BatchNormalization()(Conv_7)
    Act_Conv_7 = Activation('relu')(BN_Conv_7)
    Pool_Conv_7 = MaxPooling2D(pool_size=(2, 2))(Act_Conv_7)
    Flatten_1 = Flatten()(Pool_Conv_7)
    FC_1 = Dense(500, activation='relu')(Flatten_1)
    output_layer = Dense(2, activation='softmax')(FC_1)
    model = Model(inputs=images, outputs=output_layer)

    return model
