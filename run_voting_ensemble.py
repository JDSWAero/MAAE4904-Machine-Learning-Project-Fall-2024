import os
import numpy as np
from pandas import DataFrame, read_csv
import pickle
import time
import matplotlib.pyplot as plt

from preprocess import load_x_y, load_labels, normalize, get_PCA, apply_PCA, convert_dataset_to_grayscale
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from voting_ensemble_clf import VotingEnsembleCLF
from evaluate import get_metrics

# Create folder for datasets, model files, results
project_dir = r"C:\Users\jackw\Documents\MAAE4904\Project"

# Load datasets and labels
# Set path to folder where you have downloaded batch files
batch_dir = os.path.join(project_dir,r"cifar-10-python\cifar-10-batches-py")
X_train_path = os.path.join(project_dir,'merged_X_train_batches.csv')
y_train_path = os.path.join(project_dir,'merged_y_train_batches.csv')
X_test_path = os.path.join(project_dir,'X_test_batch.csv')
y_test_path = os.path.join(project_dir,'y_test_batch.csv')

if os.path.isfile(X_train_path) and os.path.isfile(y_train_path): # Check if training batches have already been converted and merged into a csv file
    X_train_raw = read_csv(X_train_path,index_col=0,header=0).to_numpy()
    y_train = read_csv(y_train_path,index_col=0,header=0).to_numpy()
else:
    train_batches = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    X_train_raw = None
    y_train = None
    for batch in train_batches:
        train_batch_path = os.path.join(batch_dir,batch)
        X_batch, y_batch = load_x_y(train_batch_path)
        if X_train_raw is None and y_train is None:
            X_train_raw = X_batch
            y_train = y_batch
        else:
            X_train_raw = np.concatenate([X_train_raw,X_batch],axis=0)
            y_train = np.concatenate([y_train,y_batch],axis=0)
    DataFrame(X_train_raw).to_csv(os.path.join(project_dir,'merged_X_train_batches.csv'))
    DataFrame(y_train).to_csv(os.path.join(project_dir,'merged_y_train_batches.csv'))

augmented_dataset_path = os.path.join(project_dir,'augmented_dataset','supervised_augmented.pkl')
with open(augmented_dataset_path, 'rb') as f:
    augmented_dataset = pickle.load(f)

X_train_augmented = augmented_dataset['x_train_augmented'].reshape((55000,32*32*3))
y_train_augmented = augmented_dataset['y_train_augmented']

if os.path.isfile(X_test_path) and os.path.isfile(y_test_path): # Check if test batch has already been converted into a csv file
    X_test_raw = read_csv(X_test_path,index_col=0,header=0).to_numpy()
    y_test = read_csv(y_test_path,index_col=0,header=0).to_numpy()
else:
    test_batch_path = os.path.join(batch_dir,'test_batch')
    X_batch, y_batch = load_x_y(test_batch_path)

label_names_batch_path = os.path.join(batch_dir,'batches.meta')
labels = load_labels(label_names_batch_path)

# Perform preprocessing steps
grayscale = False
if grayscale:
    X_train_grayscale_path = os.path.join(project_dir,'merged_X_train_grayscale_batches.csv')
    if os.path.isfile(X_train_grayscale_path): # Check if training batches have already been converted and merged into a csv file
        X_train_grayscale = read_csv(X_train_grayscale_path,index_col=0,header=0).to_numpy()
    else: # Switch between X_train_raw or X_train_augmented
        X_train_grayscale = convert_dataset_to_grayscale(X_train_raw)
        DataFrame(X_train_grayscale).to_csv(X_train_grayscale_path)
    X_test_grayscale = convert_dataset_to_grayscale(X_test_raw)
    X_train_norm = normalize(X_train_grayscale)
    X_test_norm = normalize(X_test_grayscale)
else: # Switch between X_train_raw or X_train_augmented
    X_train_norm = normalize(X_train_raw)
    X_test_norm = normalize(X_test_raw)

# Perform principal-component analysis
print('Performing PCA')
pca_dir = os.path.join(project_dir,'PCA')
pca = get_PCA(X_train=X_train_norm,n_components=100,pca_dir=pca_dir)
X_train_norm_transformed = apply_PCA(pca_n=pca,X=X_train_norm)
X_test_norm_transformed = apply_PCA(pca_n=pca,X=X_test_norm)

# Create and train models
# Switch between X_train_norm or X_train_norm_transformed
train_dataset = (X_train_norm,y_train)
max_iter = 50
svm_rbf_clf = SVC(kernel='rbf', C=1.0, gamma='auto', random_state=42, max_iter=max_iter, verbose=True)
svm_sigmoid_clf = SVC(kernel='sigmoid', C=1.0, gamma='auto', random_state=42, max_iter=max_iter, verbose=True)
svm_poly_clf = SVC(kernel='poly', C=1.0, gamma='auto', random_state=42, max_iter=max_iter, verbose=True)
rf_clf_100 = RandomForestClassifier(n_estimators=100, max_leaf_nodes=16, n_jobs=-1, verbose=1)
rf_clf_250 = RandomForestClassifier(n_estimators=250, max_leaf_nodes=16, n_jobs=-1, verbose=1)
rf_clf_500 = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, verbose=1)
estimators = [('svm_rbf_clf',svm_rbf_clf),
            ('svm_sigmoid_clf',svm_sigmoid_clf),
            ('svm_poly_clf_2',svm_poly_clf),
            ('rf_clf_100',rf_clf_100),
            ('rf_clf_250',rf_clf_250),
            ('rf_clf_500',rf_clf_500)]
ensemble_name = "VotingEnsemble_hard_3"
ensemble_dir = os.path.join(project_dir,ensemble_name)
if not os.path.isdir(ensemble_dir):
    os.mkdir(ensemble_dir)
ensemble = VotingEnsembleCLF(ensemble_name=ensemble_name,ensemble_dir=ensemble_dir)
print(f'Fitting {ensemble_name}')
t1 = time.time()
ensemble.fit_ensemble(estimators=estimators,train_dataset=train_dataset)
t2 = time.time()
print(f'Training time: {t2-t1}s')

# print(f'Loading {ensemble_name}')
# model_path = os.path.join(ensemble_dir,f'{ensemble_name}.pkl')
# with open(model_path, 'rb') as f:
#     ensemble = pickle.load(f)

# Generate predictions and evaluate model
print(f'Generating predictions with {ensemble_name}')
t3 = time.time()
y_train_pred = ensemble.predict(X_train_norm)
y_test_pred = ensemble.predict(X_test_norm)
t4 = time.time()
print(f'Prediction time: {t4-t3}s')
get_metrics(y_true=y_train,y_pred=y_train_pred,labels=labels,train=True,model_dir=ensemble_dir)
get_metrics(y_true=y_test,y_pred=y_test_pred,labels=labels,train=False,model_dir=ensemble_dir)
