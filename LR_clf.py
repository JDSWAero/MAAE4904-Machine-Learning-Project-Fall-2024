import os
import numpy as np
from pickle import dump

from sklearn.linear_model import LogisticRegression as LR # Change import to model you are focusing on
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV # Exhaustive grid search takes a long time, this takes the best candidates at each iteration improving speed

class Log_Reg_CLF: # Change to name of model as needed e.g. SupportVectorMachineCLF
    '''
    ModelName Classifier for Images
    '''
    def __init__(self,train_dataset:tuple,model_dir:str,model_name:str='LR_CLF_PCA_768'):
        '''
        Create instance of classifier, instance variables for training and testing features and labels,
        instance variable for path to folder that will store model files, predictions, and evaluation metrics
        and instance variable for .

        train_dataset is a tuple expected in the format of (X_train, y_train)
        '''
        self.X_train, self.y_train = train_dataset
        self.model_filepath = os.path.join(model_dir,f'{model_name}.pkl')
        self.base_model = LR(max_iter=200, random_state = 42) # Read training progress from command line

    def get_grid_search(self):
        '''
        Get exhaustive grid search cross validation object using parameter grid to find optimal hyperparameters.
        '''
        self.parameter_grid = {'solver':['newton-cg'],
                               'verbose':[1],
                               'max_iter':[1000],
                               'tol':[2e-2],
                               #'penalty':['l2'],
                               'n_jobs':[-1],
                               'C':[2.0]} # Check scikit-learn.org for your model's hyperparameters and change as needed
        self.grid_search = HalvingGridSearchCV(estimator=self.base_model,
                                            param_grid=self.parameter_grid,
                                            scoring='accuracy',
                                            random_state=42,
                                            verbose=2)
    
    def fit(self):
        '''
        Fit model using grid search cross validation and save model to pickle, .pkl, file.
        '''
        self.get_grid_search()
        self.grid_search.fit(self.X_train,self.y_train)
        self.best_model = self.grid_search.best_estimator_
        self.best_parameters = self.grid_search.best_params_
        with open(os.path.join(self.model_filepath), "wb") as f:
            dump(self.best_model, f, protocol=5)
    
    def predict(self,X:np.array):
        '''
        Generate predictions using X input features.
        '''
        y_pred = self.best_model.predict(X=X)
        
        return y_pred
