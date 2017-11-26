import keras
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Activation, Flatten, Reshape, Conv2D, MaxPooling2D, AveragePooling2D
from keras import regularizers
from keras.losses import mean_squared_error
from keras import losses
import matplotlib.patches as patches
import numpy as np
import dicom
import cv2

from utils import *
from train_cnn import run

def open_data_AE(X_fullsize, y_pred, contour_mask):
    """
    Open dataset from the output of the CNN and
    unroll it as 64*64 = vector of 4096 elements
    :param X_fullsize: full size training set (256x256)
    :param y_pred: CNN output
    :return: input AE, output
    """
    input_AE = []
    contour_experts = []
    for j in range(y_pred.shape[0]):
        in_AE = cv2.resize(compute_roi_pred(X_fullsize, y_pred, contour_mask, j, roi_shape=32)[0],(64 , 64))
        contour = cv2.resize(compute_roi_pred(X_fullsize, y_pred, contour_mask, j)[2], (64,64), interpolation = cv2.INTERSECT_NONE)
        input_AE.append(in_AE)
        contour_experts.append(contour)
    return np.array(input_AE).reshape((-1, 64*64)), np.array(contour_experts).reshape((-1, 64*64))

def customized_loss(y_true, y_pred, alpha=0.0001, beta=3):
    """
    Create a customized loss for the stacked AE.
    Linear combination of MSE and KL divergence.
    """
    #customize your own loss components
    loss1 = losses.mean_absolute_error(y_true, y_pred)
    loss2 = losses.kullback_leibler_divergence(y_true, y_pred)
    #adjust the weight between loss components
    return (alpha/2) * loss1 + beta * loss2

def model1(X_train, param_reg, get_history=False, verbose=0):
    """
    First part of the stacked AE.
    Train the AE on the ROI input images.
    :param X_train: ROI input image
    :param get_history: boolean to return the loss history
    :return: encoded ROI image
    """
    autoencoder_0 = Sequential()
    encoder_0 = Dense(input_dim=4096, units=100, kernel_regularizer=regularizers.l2(param_reg))
    decoder_0 = Dense(input_dim=100, units=4096, kernel_regularizer=regularizers.l2(param_reg))
    autoencoder_0.add(encoder_0)
    autoencoder_0.add(decoder_0)
    autoencoder_0.compile(loss= customized_loss,optimizer='adam', metrics=['accuracy'])
    h = autoencoder_0.fit(X_train, X_train, epochs=100, verbose=verbose)

    temp_0 = Sequential()
    temp_0.add(encoder_0)
    temp_0.compile(loss= customized_loss, optimizer='adam', metrics=['accuracy'])
    encoded_X = temp_0.predict(X_train, verbose=0)
    if get_history:
        return h.history['loss'], encoded_X, encoder_0
    else:
        return encoded_X, encoder_0

def model2(encoder_0, encoded_X, X_train, param_reg, get_history=False, verbose=0):
    """
    Second part of the stacked AE.
    :param X_train: encoder ROI image
    :param get_history: boolean to return the loss history
    :return: encoding layer
    """
    autoencoder_1 = Sequential()
    encoder_1 = Dense(input_dim=100, units=100, kernel_regularizer=regularizers.l2(param_reg))
    decoder_1 = Dense(input_dim=100, units=100, kernel_regularizer=regularizers.l2(param_reg))
    autoencoder_1.add(encoder_1)
    autoencoder_1.add(decoder_1)
    autoencoder_1.compile(loss= customized_loss, optimizer='adam', metrics=['accuracy'])
    h = autoencoder_1.fit(encoded_X, encoded_X, epochs=100, verbose=verbose)

    temp_0 = Sequential()
    temp_0.add(encoder_0)
    temp_0.compile(loss= customized_loss, optimizer='adam', metrics=['accuracy'])
    encoded_X = temp_0.predict(X_train, verbose=0)
    if get_history:
        return h.history['loss'], encoder_1
    else:
        return encoder_1

def model3(X_train, Y_train, encoder_0, encoder_1, init, param_reg,
           get_history=False, verbose=0):
    """
    Last part of the stacked AE.
    :param X_train: ROI input image
    :param init: set the initial kernel weights (None for uniform)
    :param get_history: boolean to return the loss history
    :return: final model
    """
    model = Sequential()
    model.add(encoder_0)
    model.add(encoder_1)
    model.add(Dense(input_dim=100, units=4096, kernel_initializer=init, kernel_regularizer=regularizers.l2(param_reg)))
    model.compile(optimizer = 'adam', loss = "MSE", metrics=['accuracy'])
    h = model.fit(X_train, Y_train, epochs=20, verbose=verbose)
    if get_history:
        return h.history['loss'], model
    else:
        return model

def run(X_fullsize, y_pred, contour_mask, X_to_pred=None, verbose=0, param_reg=3*0.001, init_3='zero'):
    """
    Full pipeline for CNN: load the dataset, train the model and predict ROIs
    :param X_fullsize: full size training set (256x256)
    :param y_pred: CNN output for training
    :param X_to_pred: input for predictions after training (X_train if not specified)
    :param verbose: int for verbose
    :return: X_train, Y_train, contours_pred
    """
    X_train, Y_train = open_data_AE(X_fullsize, y_pred, contour_mask)
    encoded_X, encoder_0 = model1(X_train, param_reg=param_reg)
    encoder_1 = model2(encoder_0, encoded_X, X_train, param_reg=param_reg)
    h, model = model3(X_train, Y_train, encoder_0, encoder_1, param_reg=param_reg, 
        init=init_3, get_history=True, verbose=verbose)
    if not X_to_pred:
        X_to_pred = X_train
    contours = model.predict(X_to_pred)
    binarys = np.array([cv2.threshold(contour, 0, 1, cv2.INTERSECT_NONE)[1].reshape((64,64)) for contour in contours])
    return X_train, Y_train, binarys
