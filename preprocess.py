'''
Contains functions for preprocessing data through data transformation e.g. PCA, normalization, and train-test splitting.
'''

import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA

def load_x_y(filepath:str) -> tuple[np.ndarray,np.ndarray]:
    '''
    Opens and returns a data batch file from the CIFAR-10 dataset as an X, y tuple. The key:value pairs are described below for
    the dictionary that is returned when loading the batch with pickle.

    Copied from https://www.cs.toronto.edu/~kriz/cifar.html:
    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
    The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
    The image is stored in row-major order, so that the first 32 entries of the array are the red channel values
    of the first row of the image.
    labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
    label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above.
    For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.'''
    with open(filepath, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X, y = dict[b'data'], dict[b'labels']

    return X, y

def load_labels(filepath:str='batches.meta') -> list:
    '''
    Opens and returns label names for CIFAR-10 dataset as a list.
    '''
    with open(filepath, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    labels = dict[b'label_names']

    return labels

def normalize(X:np.array):
    '''
    Normalize pixel values from 0-255 to 0-1 by dividing by 255.
    '''
    X_norm = X / 255

    return X_norm

def rgb_to_grayscale(X_i:np.ndarray):
    '''
    Convert a colour image to grayscale using NTSC formula: 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue.
    '''
    X_i_grayscale = np.zeros(32*32)

    for j in range(0,32*32,32):
        for i in range(32):
            r = X_i[i+j]            # Store the red channel value
            g = X_i[i+j+1023]       # Store the green channel value
            b = X_i[i+j+1023*2]     # Store the blue channel value
            gray = 0.299*r + 0.587*g + 0.114*b
            X_i_grayscale[i+j] = gray

    return X_i_grayscale

def convert_dataset_to_grayscale(X:np.ndarray):
    '''
    Convert a dataset X of coloured images to grayscale in a loop using rgb_to_grayscale.
    '''
    X_converted = np.zeros((X.shape[0],32*32))
    for i in range(X.shape[0]):
        X_i_grayscale = rgb_to_grayscale(X[i])
        X_converted[i] = X_i_grayscale
    
    return X_converted

def separate_and_reshape_channels(X_i:np.ndarray):
    '''
    Separate and reshape an instance of 32*32*3 pixel array into red, green, and blue channels.
    '''
    X_i_red = np.zeros(32*32)
    X_i_green = np.zeros(32*32)
    X_i_blue = np.zeros(32*32)

    for j in range(0,32*32,32):
        for i in range(32):
            r = X_i[i+j]            # Store the red channel value
            g = X_i[i+j+1023]       # Store the green channel value
            b = X_i[i+j+1023*2]     # Store the blue channel value
            X_i_red[i+j] = r
            X_i_green[i+j] = g
            X_i_blue[i+j] = b

    return np.moveaxis(X_i.reshape((3,32,32)),0,-1)

def get_feature_map(X_i:np.ndarray,channel_filters:np.ndarray,bias:int):
    red_filter = channel_filters[:,:,0]
    green_filter = channel_filters[:,:,1]
    blue_filter = channel_filters[:,:,2]
    
    feature_map = np.zeros((30,30))
    for i in range(1,30):
        for j in range(1,30):
            red_channel_grid = X_i[i-1:i+2,j-1:j+2,0]
            green_channel_grid = X_i[i-1:i+2,j-1:j+2,1]
            blue_channel_grid = X_i[i-1:i+2,j-1:j+2,2]
            red_product = red_channel_grid * red_filter
            green_product = green_channel_grid * green_filter
            blue_product = blue_channel_grid * blue_filter
            
            feature_map[i,j] = relu(np.sum([np.sum(red_product),np.sum(green_product),np.sum(blue_product),bias]))
    
    return feature_map.flatten()

def get_feature_map_dataset(X:np.ndarray,channel_filters:np.ndarray,bias:int):
    X_featuremap = np.zeros((X.shape[0],30*30))
    for i in tqdm(range(X.shape[0])):
        X_i = np.moveaxis(X[i,:].reshape((3,32,32)),0,-1)
        X_featuremap[i,:] = get_feature_map(X_i,channel_filters,bias)

    return X_featuremap

def get_PCA(X_train:np.ndarray,n_components:int,pca_dir:str):
    '''
    Fits principal-component analysis algorithm to the training features X_train and keeps n_components principal-components.
    '''
    if n_components == 0:
        pca_n = PCA()
        pca_n.fit(X_train)
        with open(os.path.join(pca_dir,f'pca_all.pkl'), "wb") as f:
                pickle.dump(pca_n, f, protocol=5)
    else:
        pca_n = PCA(n_components=n_components)
        pca_n.fit(X_train)
        with open(os.path.join(pca_dir,f'pca_{n_components}.pkl'), "wb") as f:
                pickle.dump(pca_n, f, protocol=5)

    return pca_n

def apply_PCA(pca_n:PCA,X:np.ndarray):
    '''
    Applies principal-component analysis to the features X and keeps n principal-components.
    '''
    X_transformed = pca_n.transform(X)

    return X_transformed

def relu(x):
    '''
    Rectified linear unit function. Returns x if greater than zero, otherwise returns 0.
    '''
    if x > 0:
        return x
    else:
        return 0
