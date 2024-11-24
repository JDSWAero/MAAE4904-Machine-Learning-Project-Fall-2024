'''
This is the main file that calls on t'''

import os
import numpy as np
from pandas import DataFrame, read_csv

from preprocess import load_x_y, load_labels, normalize, get_PCA, apply_PCA, convert_dataset_to_grayscale
from knearest_neighbours_clf import KNNCLF, EnsembleKNNCLF
from multilayer_perceptron_clf import MLPCLF, EnsembleMLPCLF
from evaluate import get_metrics

project_dir = r"C:\Users\jackw\Documents\MAAE4904\Project" # Create folder for datasets, model files, results

# Load datasets and labels
batch_dir = os.path.join(project_dir,r"cifar-10-python\cifar-10-batches-py") # Change path to where you have downloaded batch files
X_train_path = os.path.join(project_dir,'merged_X_train_batches.csv')
y_train_path = os.path.join(project_dir,'merged_y_train_batches.csv')

if os.path.isfile(X_train_path) and os.path.isfile(y_train_path): # Check if training batches have already been converted and merged into a csv file
    X_train = read_csv(X_train_path,index_col=0,header=0).to_numpy()
    y_train = read_csv(y_train_path,index_col=0,header=0).to_numpy()
else:
    train_batches = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    X_train = None
    y_train = None
    for batch in train_batches:
        train_batch_path = os.path.join(batch_dir,batch)
        X_batch, y_batch = load_x_y(train_batch_path)
        if X_train is None and y_train is None:
            X_train = X_batch
            y_train = y_batch
        else:
            X_train = np.concatenate([X_train,X_batch],axis=0)
            y_train = np.concatenate([y_train,y_batch],axis=0)
    DataFrame(X_train).to_csv(os.path.join(project_dir,'merged_X_train_batches.csv'))
    DataFrame(y_train).to_csv(os.path.join(project_dir,'merged_y_train_batches.csv'))

test_batch_path = os.path.join(batch_dir,'test_batch')
label_names_batch_path = os.path.join(batch_dir,'batches.meta')

X_test, y_test = load_x_y(test_batch_path)
labels = load_labels(label_names_batch_path)

# Perform preprocessing steps
grayscale = False
if grayscale:
    X_train_grayscale_path = os.path.join(project_dir,'merged_X_train_grayscale_batches.csv')
    if os.path.isfile(X_train_grayscale_path): # Check if training batches have already been converted and merged into a csv file
        X_train_grayscale = read_csv(X_train_grayscale_path,index_col=0,header=0).to_numpy()
    else:
        X_train_grayscale = convert_dataset_to_grayscale(X_train)
        DataFrame(X_train_grayscale).to_csv(X_train_grayscale_path)
    X_test_grayscale = convert_dataset_to_grayscale(X_test)
    X_train_norm = normalize(X_train_grayscale)
    X_test_norm = normalize(X_test_grayscale)
else:
    X_train_norm = normalize(X_train)
    X_test_norm = normalize(X_test)

# Perform principal-component analysis
pca_dir = os.path.join(project_dir,'PCA')
pca = get_PCA(X_train=X_train_norm,n_components=1536,pca_dir=pca_dir)
X_train_norm_transformed = apply_PCA(pca_n=pca,X=X_train_norm)
X_test_norm_transformed = apply_PCA(pca_n=pca,X=X_test_norm)

# Create and train models
model_name = "MLP_CLF_PCA_1536_unoptimized_1"
model_dir = os.path.join(project_dir,model_name)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
clf = MLPCLF((X_train_norm_transformed,y_train),model_dir,model_name)
clf.fit(optimize=False)

# Generate predictions and evaluate model
y_test_pred = clf.predict(X_test_norm_transformed)
get_metrics(y_true=y_test,y_pred=y_test_pred,labels=labels,train=False,model_dir=model_dir)
