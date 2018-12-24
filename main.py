import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import tensorflow as tf
from skimage import exposure
from tensorflow.contrib.layers import flatten

from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU,GlobalAveragePooling2D, regularizers
from keras.layers.convolutional import Convolution2D, Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import adam

import cv2, glob, time



def buildNetwork(X, keepProb):
    mu = 0
    sigma = 0.3

    output_depth = {
        0: 3,
        1: 8,
        2: 16,
        3: 32,
        4: 3200,
        5: 240,
        6: 120,
        7: 43,
    }

    # Layer 1: Convolutional + MaxPooling + ReLu + dropout. Input = 64x64x3. Output = 30x30x8.
    layer_1 = tf.Variable(tf.truncated_normal([5, 5, output_depth[0], output_depth[1]], mean=mu, stddev=sigma))
    layer_1 = tf.nn.conv2d(X, filter=layer_1, strides=[1, 1, 1, 1], padding='VALID')
    layer_1 = tf.add(layer_1, tf.zeros(output_depth[1]))
    layer_1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer_1 = tf.nn.dropout(layer_1, keepProb)
    layer_1 = tf.nn.relu(layer_1)

    return layer_1



# not hotdog model 1
def kerasModel(inputShape):
    model = Sequential()
    model.add(Convolution2D(8, 5, 5, border_mode='valid', input_shape=inputShape))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, 3, 3))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))

    #model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(240))
    model.add(Activation('relu'))

    model.add(Dense(120))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(2))

    model.add(Activation('softmax'))
    return model

#my model
def kerasModel3(inputShape):
    model = Sequential()
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4),border_mode='valid', input_shape=inputShape))
    #model.add(ELU())
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    #model.add(ELU())
    model.add(Activation('relu'))
    # model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    #model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    #model.add(Dropout(.2))
    #model.add(ELU())
    #model.add(Activation('relu'))
    #model.add(Dense(512,kernel_regularizer=regularizers.l2(0.00005)))

    model.add(Dense(1024))
    model.add(Dropout(.5))
    #model.add(ELU())
    model.add(Activation('relu'))

    model.add(Dense(256))
    model.add(Dropout(.5))
    #model.add(ELU())
    model.add(Activation('relu'))

    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model

#cat vs dog model
def kerasModel4(inputShape):
        model = Sequential()
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='valid', input_shape=inputShape))
        # model.add(ELU())
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 5, 5, border_mode="same"))
        # model.add(ELU())
        model.add(Activation('relu'))
        # model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
        # model.add(Flatten())
        model.add(GlobalAveragePooling2D())
        # model.add(Dropout(.2))
        # model.add(ELU())
        # model.add(Activation('relu'))
        # model.add(Dense(512,kernel_regularizer=regularizers.l2(0.00005)))

        # model.add(Dense(1024))
        # model.add(Dropout(.5))
        # # model.add(ELU())
        # model.add(Activation('relu'))

        model.add(Dense(512))
        model.add(Dropout(.5))
        # model.add(ELU())
        model.add(Activation('relu'))

        # model.add(Dense(256))
        # model.add(Dropout(.5))
        # # model.add(ELU())
        # model.add(Activation('relu'))

        model.add(Dense(3))
        model.add(Activation('softmax'))
        return model

# not hot dog model 2
def kerasModel2(inputShape):
    model = Sequential()
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4),border_mode='valid', input_shape=inputShape))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

start_time = time.time()

sizeX = 128
sizeY = 128

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print("train shape X", X_train.shape)
print("train shape y", y_train.shape)

inputShape = (sizeX, sizeY, 3)
model = kerasModel4(inputShape)

early = EarlyStopping(monitor='val_loss', patience = 10, verbose = 0)
#early = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=2, mode='auto')
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_train, y_train, nb_epoch=100, callbacks=[early],validation_split=0.1)

#a = adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.compile(loss='categorical_crossentropy',
#              optimizer=a,
#              metrics=['accuracy'])
#Callback()
#earlyStop = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=2, mode='auto')
#callbacks_list = [earlyStop]
#model.fit(X_train, y_train, batch_size=1, epochs=30, validation_data=(X_test, y_test), verbose=2, callbacks=callbacks_list)


metrics = model.evaluate(X_test, y_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

elapsed_time = time.time() - start_time
print (elapsed_time)

print("Saving model weights and configuration file")

model.save('my_model3conlayer.h5')
# serialize model to JSON
model_json = model.to_json()
with open("model3conlayer.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("modelWeight3conlayer.h5")
print("Saved model to disk")