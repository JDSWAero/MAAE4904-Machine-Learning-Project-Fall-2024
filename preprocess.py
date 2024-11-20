'''
Contains functions for preprocessing data through data transformation e.g. PCA, normalization, and train-test splitting.
'''

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split

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

batch_dir = r"C:\Users\jackw\Documents\MAAE4904\Project\cifar-10-python\cifar-10-batches-py"
train_batch_path = os.path.join(batch_dir,'data_batch_1')
test_batch_path = os.path.join(batch_dir,'test_batch')
label_names_batch_path = os.path.join(batch_dir,'batches.meta')

X_train, y_train = load_x_y(train_batch_path)
X_test, y_test = load_x_y(test_batch_path)
labels = load_labels(label_names_batch_path)
