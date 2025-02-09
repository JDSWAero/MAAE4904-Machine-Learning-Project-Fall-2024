'''
This is a template file for creating new machine learning models. Create and rename a copy of this file based on the
model you are training.
'''

import os
import numpy as np
import pickle
import json

from sklearn.ensemble import RandomForestClassifier# Change import to model you are focusing on
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV # Exhaustive grid search takes a long time, this takes the best candidates at each iteration improving speed
from sklearn.ensemble import BaggingClassifier


class RandomForestCLF: # Change to name of model as needed e.g. SupportVectorMachineCLF
    '''
    ModelName Classifier for Images
    '''
    def __init__(self,train_dataset:tuple,model_dir:str,model_name:str='RandomForestCLF'):
        '''
        Create instance of classifier, instance variables for training and testing features and labels,
        instance variable for path to folder that will store model files, predictions, and evaluation metrics
        and instance variable for .

        train_dataset is a tuple expected in the format of (X_train, y_train)
        '''
        self.X_train, self.y_train = train_dataset
        self.y_train = self.y_train.ravel()
        self.model_path = os.path.join(model_dir,f'{model_name}.pkl')
        self.model_hyperparameters_path = os.path.join(model_dir,f'{model_name}_hyperparameters.json')
        self.base_model = RandomForestClassifier(max_leaf_nodes=16, n_jobs=-1) # Read training progress from command line

    def get_grid_search(self):
        '''
        Get exhaustive grid search cross validation object using parameter grid to find optimal hyperparameters.
        '''
        self.parameter_grid = {'n_estimators':[500],
                               'max_depth':[60],
                               'min_samples_split':[2],
                               'min_samples_leaf':[2]
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
                pickle.dump(self.model, f, protocol=5)
            with open(os.path.join(self.model_hyperparameters_path), "w", encoding='utf-8') as f:
                json.dump(self.hyperparameters, f)
        else:
            self.model = self.base_model
            self.model.fit(self.X_train,self.y_train)
            self.hyperparameters = self.model.get_params()
            with open(os.path.join(self.model_path), "wb") as f:
                pickle.dump(self.model, f, protocol=5)
            with open(os.path.join(self.model_hyperparameters_path), "w", encoding='utf-8') as f:
                json.dump(self.hyperparameters, f, ensure_ascii=False, indent=4)
    
    def predict(self,X:np.array):
        '''
        Generate predictions using X input features.
        '''
        y_pred = self.model.predict(X=X)
        
        return y_pred


class EnsembleModelNameCLF:
    def __init__(self,n_estimators:int,train_dataset:tuple,model_dir:str,model_name:str='EnsembleModelNameCLF'):
        '''
        Create instance of classifier, instance variables for training and testing features and labels,
        instance variable for path to folder that will store model files, predictions, and evaluation metrics
        and instance variable for .

        train_dataset is a tuple expected in the format of (X_train, y_train)
        '''
        self.X_train, self.y_train = train_dataset
        self.model_filepath = os.path.join(model_dir,f'{model_name}.pkl')
        self.base_model = RandomForestClassifier(n_estimators=25,max_leaf_nodes=16, n_jobs=-1) # Read training progress from command line
        self.ensemble_model = BaggingClassifier(estimator=self.base_model,n_estimators=n_estimators,random_state=42,verbose=1)

    def fit(self):
        '''
        Fit model using grid search cross validation and save model to pickle, .pkl, file.
        '''
        self.ensemble_model.fit(self.X_train,self.y_train)
        with open(os.path.join(self.model_filepath), "wb") as f:
            json.dump(self.ensemble_model, f, protocol=5)
    
    def predict(self,X:np.array):
        '''
        Generate predictions using X input features.
        '''
        y_pred = self.ensemble_model.predict(X=X)
        
        return y_pred
