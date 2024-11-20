'''
This is a template file for creating new machine learning models. Create and rename a copy of this file based on the
model you are training.
'''

import os
import numpy as np
from pickle import dump

from sklearn.svm import SVC # Change import to model you are focusing on
from sklearn.model_selection import GridSearchCV

class ModelNameCLF: # Change to name of model as needed e.g. SupportVectorMachineCLF
    '''
    ModelName Classifier for Images
    '''
    def __init__(self,train_dataset:tuple,model_dir:str,model_file:str='ModelNameCLF.pkl'):
        '''
        Create instance of classifier, instance variables for training and testing features and labels,
        instance variable for path to folder that will store model files, predictions, and evaluation metrics
        and instance variable for .

        train_dataset is a tuple expected in the format of (X_train, y_train)
        '''
        self.X_train, self.y_train = train_dataset
        self.model_filepath = os.path.join(model_dir,model_file)
        self.base_model = SVC(verbose=True) # Read training progress from command line
        self.get_grid_search()
        self.fit()

    def get_grid_search(self):
        '''
        Get exhaustive grid search cross validation object using parameter grid to find optimal hyperparameters.
        '''
        self.parameter_grid = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                               'degree':[2,3,4],
                               'gamma':['scale','auto']} # Check scikit-learn.org for your model's hyperparameters and change as needed
        self.grid_search = GridSearchCV(estimator=self.base_model,
                                            param_grid=self.parameter_grid,
                                            scoring='accuracy',
                                            verbose=2)
    
    def fit(self):
        '''
        Fit model using grid search cross validation and save model to pickle, .pkl, file.
        '''
        self.grid_search.fit(self.X_train,self.y_train)
        self.best_model = self.grid_search.best_estimator_
        self.best_parameters = self.grid_search.best_params_
        with open(os.path.join(self.model_filepath), "wb") as f:
            dump(self.best_model, f, protocol=5)
    
    def predict(self,X:np.array):
        y_pred = self.best_model.predict(X=X)
        
        return y_pred
