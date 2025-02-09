'''
Multi-layer Perceptron Image Classifier

Jack Wooldridge, 101181465
'''

import os
import numpy as np
import pickle
import json

from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import BaggingClassifier

class MLPCLF:
    '''
    ModelName Classifier for Images
    '''
    def __init__(self,train_dataset:tuple,model_dir:str,model_name:str='MLPCLF'):
        '''
        Create instance of classifier, instance variables for training and testing features and labels,
        instance variable for path to folder that will store model files, predictions, and evaluation metrics
        and instance variable for model name.

        train_dataset is a tuple expected in the format of (X_train, y_train)
        '''
        self.X_train, self.y_train = train_dataset
        self.model_path = os.path.join(model_dir,f'{model_name}.pkl')
        self.hyperparameters_path = os.path.join(model_dir,f'{model_name}_hyperparameters.json')
        self.base_model = MLPClassifier(hidden_layer_sizes=(75,50),
                                        activation='relu',
                                        solver='sgd',
                                        random_state=42,
                                        tol=0.001,
                                        learning_rate_init=0.005,
                                        early_stopping=True,
                                        verbose=True,
                                        n_iter_no_change=10,
                                        batch_size=1000,
                                        max_iter=100)

    def get_grid_search(self):
        '''
        Get exhaustive grid search cross validation object using parameter grid to find optimal hyperparameters.
        '''
        self.parameter_grid = {'activation':['relu'],
                            'solver':['adam'],
                            'batch_size':[200,500,1000]}
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
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f, protocol=5)
            with open(self.hyperparameters_path, "w", encoding='utf-8') as f:
                json.dump(self.hyperparameters, f)
        else:
            self.model = self.base_model
            self.model.fit(self.X_train,self.y_train)
            self.hyperparameters = self.model.get_params()
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f, protocol=5)
            with open(self.hyperparameters_path, "w", encoding='utf-8') as f:
                json.dump(self.hyperparameters, f)
    
    def predict(self,X:np.array):
        '''
        Generate predictions using X input features.
        '''
        y_pred = self.model.predict(X=X)
        
        return y_pred

class EnsembleMLPCLF:
    def __init__(self,n_estimators:int,train_dataset:tuple,model_dir:str,model_name:str='BaggingMLPCLF'):
        '''
        Create instance of classifier, instance variables for training and testing features and labels,
        instance variable for path to folder that will store model files, predictions, and evaluation metrics
        and instance variable for .

        train_dataset is a tuple expected in the format of (X_train, y_train)
        '''
        self.X_train, self.y_train = train_dataset
        self.model_path = os.path.join(model_dir,f'{model_name}.pkl')
        self.hyperparameters_path = os.path.join(model_dir,f'{model_name}_hyperparameters.json')
        self.base_model = MLPClassifier(hidden_layer_sizes=(1024,512,512,256)) # Read training progress from command line
        self.ensemble_model = BaggingClassifier(estimator=self.base_model,
                                                n_estimators=n_estimators,
                                                verbose=2)

    def fit(self):
        '''
        Fit model using grid search cross validation and save model to pickle, .pkl, file.
        '''
        self.ensemble_model.fit(self.X_train,self.y_train)
        self.hyperparameters = self.ensemble_model.get_params()
        with open(self.model_path, "wb") as f:
            pickle.dump(self.ensemble_model, f, protocol=5)
        with open(self.hyperparameters_path, "w", encoding='utf-8') as f:
            json.dump(self.hyperparameters, f)
    
    def predict(self,X:np.array):
        '''
        Generate predictions using X input features.
        '''
        y_pred = self.ensemble_model.predict(X=X)
        
        return y_pred
