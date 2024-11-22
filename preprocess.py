'''
Contains functions for preprocessing data through data transformation e.g. PCA, normalization, and train-test splitting.
'''

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA

def load_x_y(filepath:str):
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

def load_labels(filepath:str='batches.meta'):
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

def get_PCA(X_train:np.ndarray,n_components:int,pca_dir:str):
    '''
    Fits principal-component analysis algorithm to the training features X_train and keeps n_components principal-components.
    '''
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
