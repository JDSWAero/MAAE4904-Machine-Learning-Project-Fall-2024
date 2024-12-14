'''
Contains functions for calculating and saving metrics to evaluate machine learning models.
'''

import os
import numpy as np
from pandas import DataFrame

from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_metrics(y_true:np.array,y_pred:np.array,labels:list,train:bool,model_dir:str):
    '''
    Calculate and save accuracy score and mean squared error,
    and generate and save confusion matrix for multiclass predictions.
    '''
    if train:
        train_or_test = 'train'
    else:
        train_or_test = 'test'
    accuracy = accuracy_score(y_true=y_true,y_pred=y_pred)
    precision = precision_score(y_true=y_true,y_pred=y_pred,average=None)
    recall = recall_score(y_true=y_true,y_pred=y_pred,average=None)
    print(f'{train_or_test} accuracy: {accuracy}')
    print(f'{train_or_test} precision: {precision}')
    print(f'{train_or_test} recall: {recall}')
    metrics_df = DataFrame.from_dict({'precision':precision,'recall':recall})
    metrics_df.to_csv(os.path.join(model_dir,f'{train_or_test}_metrics.csv'))
    
    cm = confusion_matrix(y_true=y_true,y_pred=y_pred)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    cm_disp.plot()
    plt.savefig(os.path.join(model_dir,f'{train_or_test}_confusion_matrix.png'))
    plt.show()
    plt.close()
