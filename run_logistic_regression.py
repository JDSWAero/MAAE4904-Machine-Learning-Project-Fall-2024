import os
import numpy as np
from pandas import DataFrame, read_csv
from sklearn.decomposition import PCA
import pickle

from preprocess import load_x_y, load_labels, normalize, get_PCA, apply_PCA, __init__
from LR_clf import Log_Reg_CLF
from evaluate import get_metrics

# Create variable for path to folder containing datasets, model files, and results
project_dir = r"C:\Users\bradl\OneDrive\Desktop\MAAE4904\MAAE4904-Machine-Learning-Project-Fall-2024"

# Load datasets and labels
# Set path to folder where you have downloaded batch files
batch_dir = os.path.join(project_dir,r"cifar-10-python\cifar-10-batches-py")
X_train_path = os.path.join(project_dir,'merged_X_train_batches.csv')
y_train_path = os.path.join(project_dir,'merged_y_train_batches.csv')

with open('supervised_augmented.pkl', 'rb') as f:
    data = pickle.load(f)

if 'x_train_augmented' in data and 'y_train_augmented' in data:
        # For supervised_augmented.pkl
        x_data = data['x_train_augmented']
        y_labels = data['y_train_augmented']

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
X_train_norm = normalize(X_train)
X_test_norm = normalize(X_test)
pca_dir = os.path.join(project_dir,'PCA')
pca = get_PCA(X_train=X_train_norm,n_components=768,pca_dir=pca_dir) # 80% of pixels
X_train_norm_transformed = apply_PCA(pca_n=pca,X=X_train_norm)
X_test_norm_transformed = apply_PCA(pca_n=pca,X=X_test_norm)

X_data_norm = normalize(x_data)

# Create and train models
model_name = "LR_CLF_PCA_768"
model_dir = os.path.join(project_dir,model_name)
if  not os.path.isdir(model_dir):
    os.mkdir(model_dir)
#log_reg_clf = Log_Reg_CLF((X_train_norm,y_train),model_dir,model_name)
log_reg_clf = Log_Reg_CLF((x_data, y_labels),model_dir,model_name)
log_reg_clf.fit()

# Generate predictions and evaluate model
y_test_pred = log_reg_clf.predict(X_test)
get_metrics(y_true=y_test,y_pred=y_test_pred,labels=labels,train=False,model_dir=model_dir)  
