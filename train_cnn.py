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

def create_model(activation, input_shape=(64, 64)):
    """
    Simple convnet model : one convolution, one average pooling and one fully connected layer
    :param activation: None if nothing passed, e.g : ReLu, tanh, etc.
    :return: Keras model
    """
    model = Sequential()
    model.add(Conv2D(100, (11,11), activation=activation, padding='valid', strides=(1, 1), input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(AveragePooling2D((6,6)))
    model.add(Reshape([-1, 8100]))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Reshape([-1, 32, 32]))
    return model

def create_model_maxpooling(activation, input_shape=(64, 64)):
    """
    Simple convnet model with max pooling: one convolution, one max pooling and one fully connected layer
    :param activation: None if nothing passed, e.g : ReLu, tanh, etc.
    :return: Keras model
    """
    model = Sequential()
    model.add(Conv2D(100, (11,11), activation=activation, padding='valid', strides=(1, 1), input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(MaxPooling2D((6,6)))
    model.add(Reshape([-1, 8100]))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Reshape([-1, 32, 32]))
    return model

def create_model_larger(activation, input_shape=(64, 64)):
    """
    Larger (more filters) convnet model : one convolution, one average pooling and one fully connected layer:
    :param activation: None if nothing passed, e.g : ReLu, tanh, etc. 
    :return: Keras model
    """
    model = Sequential()
    model.add(Conv2D(200, (11,11), activation=activation, padding='valid', strides=(1, 1), input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(AveragePooling2D((6,6)))
    model.add(Reshape([-1, 16200]))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Reshape([-1, 32, 32]))
    return model

def create_model_deeper(activation, input_shape=(64, 64)):
    """
    Deeper convnet model : two convolutions, two average pooling and one fully connected layer:
    :param activation: None if nothing passed, e.g : ReLu, tanh, etc.
    :return: Keras model
    """
    model = Sequential()
    model.add(Conv2D(64, (11,11), activation=activation, padding='valid', strides=(1, 1), input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(AveragePooling2D((2,2)))
    model.add(Conv2D(128, (10, 10), activation=activation, padding='valid', strides=(1, 1)))
    model.add(AveragePooling2D((2,2)))
    model.add(Reshape([-1, 128*9*9]))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Reshape([-1, 32, 32]))
    return model

def create_model_full(activation, input_shape=(64, 64)):
    model = Sequential()
    model.add(Conv2D(64, (11,11), activation=activation, padding='valid', strides=(1, 1), input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (10, 10), activation=activation, padding='valid', strides=(1, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Reshape([-1, 128*9*9]))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Reshape([-1, 32, 32]))
    return model

def training(m, X, Y, verbose, batch_size=16, epochs=20, data_augm=False):
    """
    Training CNN with the possibility to use data augmentation
    :param m: Keras model
    :param epochs: number of epochs
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
    return history, m

def run(model='simple', X_to_pred=None, history=False, verbose=0, activation=None, epochs=20, data_augm=False):
    """
    Full pipeline for CNN: load the dataset, train the model and predict ROIs
    :param model: choice between different models e.g simple, larger, deeper, maxpooling
    :param activation: None if nothing passed, e.g : ReLu, tanh, etc.
    :param epochs: number of epochs
    :param X_to_pred: input for predictions after training (X_train if not specified)
    :param verbose: int for verbose
    :return: X, X_fullsize, Y, y_pred, h (if history boolean passed)
    """
    X, X_fullsize, Y, contour_mask = create_dataset()
    if model == 'simple':
        m = create_model(activation=activation)
    elif model == 'larger':
        m = create_model_larger(activation=activation)
    elif model == 'deeper':
        m = create_model_deeper(activation=activation)
    elif model == 'maxpooling':
        m = create_model_maxpooling(activation=activation)
    elif model =='full':
        m = create_model_full(activation=activation)

    m.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    if verbose > 0:
        print('Size for each layer :\nLayer, Input Size, Output Size')
        for p in m.layers:
            print(p.name.title(), p.input_shape, p.output_shape)
    h, m = training(m, X, Y, verbose=verbose, batch_size=16, epochs=epochs, data_augm=data_augm)

    if not X_to_pred:
        X_to_pred = X
    y_pred = m.predict(X_to_pred, batch_size=16)
    
    if history:
        return X, X_fullsize, Y, contour_mask, y_pred, h, m
    else:
        return X, X_fullsize, Y, contour_mask, y_pred, m

def inference(model):
    X_test, X_fullsize_test, Y_test, contour_mask_test = create_dataset(n_set='test')
    y_pred = model.predict(X_test, batch_size=16)
    return X_test, X_fullsize_test, Y_test, contour_mask_test, y_pred
