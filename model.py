from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Flatten, Dropout


# Create a basic deep learning model to be used as a classifier
def create_deep_model(input_shape):

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

