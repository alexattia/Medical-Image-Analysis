import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.losses import mean_squared_error
import glob
import matplotlib.patches as patches
import json
import numpy as np
from matplotlib.path import Path
import dicom
import cv2

from utils import *

def create_model(input_shape=(64, 64)):
    """
    Simple convnet model : one convolution, one average pooling and one fully connected layer:
    :return: Keras model
    """
    model = Sequential()
    model.add(Conv2D(100, (11,11), padding='valid', strides=(1, 1), input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(AveragePooling2D((6,6)))
    model.add(Reshape([-1, 8100]))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Reshape([-1, 32, 32]))
    return model

def training(m, X, Y, verbose, batch_size=16, epochs= 10, data_augm=False):
    """
    Training CNN with the possibility to use data augmentation
    :param m: Keras model
    :param X: training pictures
    :param Y: training binary ROI mask
    :return: history
    """
    if data_augm:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=50,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False) 
        datagen.fit(X)
        history = m.fit_generator(datagen.flow(X, Y,
                                    batch_size=batch_size),
                                    steps_per_epoch=X.shape[0] // batch_size,
                                    epochs=epochs,
                                    verbose=verbose)         
    else:
        history = m.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=verbose)
    return history

def run(X_to_pred=None, verbose=0):
    """
    Full pipeline for CNN: load the dataset, train the model and predict ROIs
    :param X_to_pred: input for predictions after training (X_train if not specified)
    :param verbose: int for verbose
    :return: X, X_fullsize, Y, y_pred
    """
    X, X_fullsize, Y, contour_mask = create_dataset()
    m = create_model()
    m.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    if verbose > 0:
        print('Size for each layer :\nLayer, Input Size, Output Size')
        for p in m.layers:
            print(p.name.title(), p.input_shape, p.output_shape)
    h = training(m, X, Y, verbose=verbose, batch_size=16, epochs=20, data_augm=False)

    if not X_to_pred:
        X_to_pred = X
    y_pred = m.predict(X_to_pred, batch_size=16)
    return X, X_fullsize, Y, contour_mask, y_pred