import os
import numpy as np
from pandas import DataFrame, read_csv
import pickle

from preprocess import load_x_y, load_labels, normalize, get_PCA, apply_PCA, convert_dataset_to_grayscale
from linearsvm_clf import LinearSVMCLF, EnsembleLinearSVMCLF
from nonlinearsvm_clf import NonLinearSVMCLF, EnsembleNonLinearSVMCLF
from evaluate import get_metrics

# Create variable for path to folder containing datasets, model files, and results
project_dir = r"C:\Users\jackw\Documents\MAAE4904\Project"

# Load datasets and labels
# Create variable for path to folder containing datasets, model files, and results
batch_dir = os.path.join(project_dir,r"cifar-10-python\cifar-10-batches-py")
X_train_path = os.path.join(project_dir,'merged_X_train_batches.csv')
y_train_path = os.path.join(project_dir,'merged_y_train_batches.csv')

# Set path to folder where you have downloaded batch files
if os.path.isfile(X_train_path) and os.path.isfile(y_train_path):
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
pca = get_PCA(X_train=X_train_norm,n_components=1536,pca_dir=pca_dir) # 75% of pixels
X_train_norm_transformed = apply_PCA(pca_n=pca,X=X_train_norm)
X_test_norm_transformed = apply_PCA(pca_n=pca,X=X_test_norm)

# Augmented dataset
augmented_filepath = os.path.join(project_dir,'train_augmented.pkl')

with open(augmented_filepath,'rb') as f:
    data = pickle.load(f)

X_train_augmented , y_train_augmented = data['x_train_augmented'], data['y_train_augmented']

X_train_aug_sep = X_train_augmented[50000:]
y_train_aug_sep = y_train_augmented[50000:]

X_train_augmented_normalize = X_train_augmented.reshape(55000, -1) / 255
X_train_augmented_flat = X_train_augmented.reshape(X_train_augmented.shape[0], -1)

X_train_aug_reduced = apply_PCA(pca_n=pca,X=X_train_augmented_flat)

X_combined = np.concatenate([X_train_aug_reduced,X_train_norm_transformed],axis=0)
y_combined = np.concatenate([y_train_augmented,y_train],axis=0)

# Create and train models
model_name = "NonLinearSVMCLF"
model_dir = os.path.join(project_dir,model_name)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
clf = NonLinearSVMCLF((X_combined,y_combined),model_dir,model_name)
clf.fit()

# Generate predictions and evaluate model
y_test_pred = clf.predict(X_test_norm_transformed)
get_metrics(y_true=y_test,y_pred=y_test_pred,labels=labels,train=False,model_dir=model_dir)


