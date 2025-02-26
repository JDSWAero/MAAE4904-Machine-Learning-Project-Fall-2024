'''
This is a template file for creating new machine learning models. Create and rename a copy of this file based on the
model you are training.
'''

import os
import numpy as np
from pickle import dump
import json

from sklearn.neighbors import KNeighborsClassifier # Change import to model you are focusing on
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV # Exhaustive grid search takes a long time, this takes the best candidates at each iteration improving speed
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_openml
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

class NonLinearSVMCLF: # Change to name of model as needed e.g. SupportVectorMachineCLF
    '''
    ModelName Classifier for Images
    '''
    def __init__(self,train_dataset:tuple,model_dir:str,model_name:str='NonLinearSVMCLF'):
        '''
        Create instance of classifier, instance variables for training and testing features and labels,
        instance variable for path to folder that will store model files, predictions, and evaluation metrics
        and instance variable for .

        train_dataset is a tuple expected in the format of (X_train, y_train)
        '''
        self.X_train, self.y_train = train_dataset
        self.model_path = os.path.join(model_dir,f'{model_name}.pkl')
        self.model_hyperparameters_path = os.path.join(model_dir,f'{model_name}_hyperparameters.json')
        self.base_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, max_iter=200) # Read training progress from command line

    def get_grid_search(self):
        '''
        Get exhaustive grid search cross validation object using parameter grid to find optimal hyperparameters.
        '''
        self.parameter_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.001],
            'kernel': ['rbf']
        }
        self.grid_search = HalvingGridSearchCV(estimator=self.base_model,
                                            param_grid=self.parameter_grid,
                                            scoring='accuracy',
                                            random_state=42,
                                            verbose=2)
    
    def fit(self,optimize=True):
        '''
        Fit model using grid search cross validation and save model to pickle, .pkl, file.
        '''
        if optimize:
            self.get_grid_search()
            self.grid_search.fit(self.X_train,self.y_train)
            self.model = self.grid_search.best_estimator_
            self.hyperparameters = self.grid_search.best_params_
            with open(os.path.join(self.model_path), "wb") as f:
                dump(self.model, f, protocol=5)
            with open(os.path.join(self.model_hyperparameters_path), "wb") as f:
                dump(self.hyperparameters, f)
        else:
            self.model = self.base_model
            self.model.fit(self.X_train,self.y_train)
            self.hyperparameters = self.model.get_params()
            with open(os.path.join(self.model_path), "wb") as f:
                dump(self.model, f, protocol=5)
            with open(os.path.join(self.model_hyperparameters_path), "wb") as f:
                dump(self.hyperparameters, f)
    
    def predict(self,X:np.array):
        '''
        Generate predictions using X input features.
        '''
        y_pred = self.model.predict(X=X)
        
        return y_pred

class EnsembleNonLinearSVMCLF:
    def __init__(self,n_estimators:int,train_dataset:tuple,model_dir:str,model_name:str='EnsembleNonLinearSVMCLF'):
        '''
        Create instance of classifier, instance variables for training and testing features and labels,
        instance variable for path to folder that will store model files, predictions, and evaluation metrics
        and instance variable for .

        train_dataset is a tuple expected in the format of (X_train, y_train)
        '''
        self.X_train, self.y_train = train_dataset
        self.model_filepath = os.path.join(model_dir,f'{model_name}.pkl')
        self.base_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42) # Read training progress from command line
        self.ensemble_model = BaggingClassifier(estimator=self.base_model,n_estimators=n_estimators,random_state=42,verbose=1)

    def fit(self):
        '''
        Fit model using grid search cross validation and save model to pickle, .pkl, file.
        '''
        self.ensemble_model.fit(self.X_train,self.y_train)
        with open(os.path.join(self.model_filepath), "wb") as f:
            dump(self.ensemble_model, f, protocol=5)
    
    def predict(self,X:np.array):
        '''
        Generate predictions using X input features.
        '''
        y_pred = self.ensemble_model.predict(X=X)
        
        return y_pred
