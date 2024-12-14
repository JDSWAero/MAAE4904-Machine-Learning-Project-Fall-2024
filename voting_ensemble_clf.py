'''
Creates voting ensemble from pre-trained classifiers.


Author: Jack Wooldridge
Last updated: 2024/12/10
'''

import os
from sklearn.ensemble import VotingClassifier
import pickle
import json
import numpy as np

class VotingEnsembleCLF:
    '''
    Voting ensemble classifier for images. Loads models and groups them to make new predictions based on voting.
    Can be built with pre-trained models or fit new models.
    '''
    def __init__(self,ensemble_name:str,ensemble_dir:str):
        '''
        Create instance of ensemble, instance variable for path to folder that will store model files, predictions,
        and evaluation metrics.
        '''
        self.ensemble_path = os.path.join(ensemble_dir,f'{ensemble_name}.pkl')

    def load_ensemble(self,model_names_paths:dict[str:str]):
        '''
        Load previously trained models to group into an ensemble.
        '''
        self.estimators = []
        for model_name,model_path in model_names_paths:
            with open(model_path, 'rb') as f:
                self.estimators.append((model_name,pickle.load(f)))
        self.ensemble = VotingClassifier(estimators=self.estimators,voting='hard',verbose=True)
        with open(self.ensemble_path, "wb") as f:
            pickle.dump(self.ensemble, f, protocol=5)
    
    def fit_ensemble(self,estimators:list,train_dataset:tuple):
        '''
        Fit new ensemble given initialized models and training dataset.
        '''
        self.estimators = estimators
        self.ensemble = VotingClassifier(estimators=self.estimators,voting='hard',verbose=True)
        X_train, y_train = train_dataset
        self.ensemble.fit(X_train,y_train)
        with open(self.ensemble_path, "wb") as f:
            pickle.dump(self.ensemble, f, protocol=5)

    def predict(self,X:np.array):
        '''
        Generate predictions using X input features.
        '''
        y_pred = self.ensemble.predict(X=X)
        
        return y_pred
