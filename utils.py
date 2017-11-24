import glob
import matplotlib.patches as patches
import json
import numpy as np
from matplotlib.path import Path
import dicom
import cv2

def get_roi(image, contour, shape_out = 32):
    """
    Create a binary mask with ROI from contour. 
    Extract the maximum square around the contour.
    :param image: input image (needed for shape only)
    :param contour: numpy array contour (d, 2)
    :return: numpy array mask ROI (shape_out, shape_out)
    """
    X_min, Y_min = contour[:,0].min(), contour[:,1].min()
    X_max, Y_max = contour[:,0].max(), contour[:,1].max()  
    w = X_max - X_min
    h = Y_max - Y_min
    mask_roi = np.zeros(image.shape)
    if w > h :
        mask_roi[int(Y_min - (w -h)/2):int(Y_max + (w -h)/2), int(X_min):int(X_max)] = 1.0
    else :
        mask_roi[int(Y_min):int(Y_max), int(X_min - (h-w)/2):int(X_max + (h -w)/2)] = 1.0
    return cv2.resize(mask_roi, (shape_out, shape_out), interpolation = cv2.INTER_NEAREST)

def create_dataset(image_shape=64, original_image_shape=256, 
                   roi_shape=32, data_path='./Data/'):
    """
    Creating the dataset from the images and the contour for the CNN.
    :param image_shape: image dataset desired size
    :param original_image_shape: original image size
    :param roi_shape: binary ROI mask shape
    :param data_path: path for the dataset
    :return: correct size image dataset, full size image dataset, label (contours) dataset
    """
    # Create dataset
    series = json.load(open('series_case.json')) 
    images, images_fullsize, contours, contour_mask = [], [], [], []
    # Loop over the series
    for case, serie in series.items():
        image_path_base = data_path + 'challenge_training/%s/IM-%s' % (case, serie)
        contour_path_base = data_path + 'Sunnybrook Cardiac MR Database ContoursPart3/\
TrainingDataContours/%s/contours-manual/IRCCI-expert/' % case
        contours_list = glob.glob(contour_path_base + '*')
        contours_list_series = [k.split('/')[7].split('-')[2] for k in contours_list]
        # Loop over the contours/images
        for c in contours_list_series:
            # Get contours and images path
            idx_contour = contours_list_series.index(c)
            image_path = image_path_base + '-%s.dcm' % c
            contour_path = contours_list[idx_contour]

            # open image as numpy array and resize to (image_shape, image_shape)
            image_part = dicom.read_file(image_path).pixel_array  

            # open contours as numpy array
            contour = []
            file = open(contour_path, 'r') 
            for line in file: 
                contour.append(tuple(map(float, line.split())))
            contour = np.array(contour)
            # append binary ROI mask 
            contours.append(get_roi(image_part, contour))

            # create mask contour with experts contours
            x, y = np.meshgrid(np.arange(256), np.arange(256)) # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x,y)).T 
            p = Path(contour) # make a polygon
            grid = p.contains_points(points)
            mask_contour = grid.reshape(256,256)
            mask_contour=mask_contour*1
            contour_mask.append(mask_contour)
            
            # Open image and resize it 
            images.append(cv2.resize(image_part, (image_shape, image_shape)))
            images_fullsize.append(cv2.resize(image_part, (original_image_shape, original_image_shape)))
    X_fullsize = np.array(images_fullsize)
    X = np.reshape(np.array(images), [len(images), image_shape, image_shape, 1])
    Y = np.reshape(np.array(contours), [len(contours), 1, roi_shape, roi_shape])
    print('Dataset shape :', X.shape, Y.shape)
    return X, X_fullsize, Y, contour_mask

def compute_roi_pred(X_fullsize, y_pred, contour_mask, idx, roi_shape=32):
    """
    Computing and cropping a ROI from the original image for further processing in the next stage
    :param X_fullsize: full size training set (256x256)
    :param y_pred: predictions
    :param contour_mask label: (contours) dataset
    :param idx: desired image prediction index
    :param roi_shape: shape of the binary mask
    """
    # up sampling from 32x32 to original MR size
    pred = cv2.resize(y_pred[idx].reshape((roi_shape, roi_shape)), (256,256), cv2.INTER_NEAREST)
    # select the non null pixels
    pos_pred = np.array(np.where(pred > 0.5))
    # get the center of the mask
    X_min, Y_min = pos_pred[0, :].min(), pos_pred[1, :].min()
    X_max, Y_max = pos_pred[0, :].max(), pos_pred[1, :].max()  
    X_middle = X_min + (X_max - X_min) / 2
    Y_middle = Y_min + (Y_max - Y_min) / 2
    # Find ROI coordinates
    X_top = int(X_middle - 50)
    Y_top = int(Y_middle - 50)
    X_down = int(X_middle + 50)
    Y_down = int(Y_middle + 50)
    # crop ROI of size 100x100
    mask_roi = np.zeros((256, 256))
    mask_roi = cv2.rectangle(mask_roi, (X_top, Y_top), (X_down, Y_down), 1, -1)*255
    return X_fullsize[idx][X_top:X_down, Y_top:Y_down], mask_roi, contour_mask[idx][X_top:X_down, Y_top:Y_down]

def prediction_plot(X, model, idx=None):
    """
    Compute the Inferred shape binary mask using the trained stacked AE model
    :param X: dataset to predict
    :param model: trained AE model
    :param idx: index of the particular picture to return
    :return: inferred shape binary mask, infered shape on the MR image
    """
    if not idx:
        idx= np.random.randint(len(X))
    contours = model.predict(X)
    contour = contours[idx].reshape((64,64))
    # thresholding
    binary = cv2.threshold(contour, 0, 1, cv2.INTERSECT_NONE)
    return binary[1], binary[1]*X[idx].reshape(64,64), idx

def dice_metric(X, Y):
    """
    Dice metric for measuring the contour overlap 
    :param X,Y: 2D numpy arrays
    :return: metric scalar
    """
    return np.sum(X[Y==1])*2.0 / (np.sum(X) + np.sum(Y))

def conformity_coefficient(X, Y):
    """
    Conformity coefficient for measuring the  ratio of the number of mis-segmented pixels  
    :param X,Y: 2D numpy arrays
    :return: metric scalar
    """
    return (3*dice_metric(X,Y)-2)/dice_metric(X,Y)