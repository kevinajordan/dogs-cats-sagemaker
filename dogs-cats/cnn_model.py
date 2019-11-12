import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Batch
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam



def build_model():
    '''
    build's a 4 layer CNN using the Keras sequential API
    '''

    model = Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=.001), metrics=['acc'] )
    return model