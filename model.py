from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Flatten, Dropout
from keras.initializers import glorot_normal

from keras.regularizers import l2

from keras.layers import Conv2DTranspose, Input, concatenate
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2
from sklearn.svm import LinearSVC


def create_deep_model(input_shape):


    num_inputs = input_shape[0]

    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))


    model.add(Dense(2))
    model.add(Activation('softmax'))


    return model

