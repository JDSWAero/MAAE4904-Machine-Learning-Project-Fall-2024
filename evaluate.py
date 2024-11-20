'''
Contains functions for calculating and saving metrics to evaluate machine learning models.
'''

import os
import numpy as np
from pandas import DataFrame

from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.pyplot import savefig

def get_metrics(y_true:np.array,y_pred:np.array,labels:list,train:bool,model_dir:str):
    '''
    Calculate and save accuracy score and mean squared error,
    and generate and save confusion matrix for multiclass predictions.
    '''
    acc = accuracy_score(y_true=y_true,y_pred=y_pred)
    print(acc)
    # mse = mean_squared_error(y_true=y_true,y_pred=y_pred)
    # metrics_df = DataFrame({'accuracy':acc,'mean_squared_error':mse})
    cm = confusion_matrix(y_true=y_true,y_pred=y_pred)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    cm_disp.plot()

    if train:
        train_or_test = 'train'
    else:
        train_or_test = 'test'
    # metrics_df.to_csv(os.path.join(model_dir,f'{train_or_test}_metrics.csv'))
    savefig(os.path.join(model_dir,f'{train_or_test}_confusion_matrix.png'))
