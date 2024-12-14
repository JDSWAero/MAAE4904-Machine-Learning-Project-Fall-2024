import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

import preprocess
from multilayer_perceptron_clf import MLPCLF
import evaluate

project_dir = r"C:\Users\jackw\Documents\MAAE4904\Project"
batch_dir = os.path.join(project_dir,r"cifar-10-python\cifar-10-batches-py")
X_train_path = os.path.join(project_dir,'merged_X_train_batches.csv')
y_train_path = os.path.join(project_dir,'merged_y_train_batches.csv')
X_test_path = os.path.join(project_dir,'X_test_batch.csv')
y_test_path = os.path.join(project_dir,'y_test_batch.csv')
X_train_raw = pd.read_csv(X_train_path,index_col=0,header=0).to_numpy()
y_train = pd.read_csv(y_train_path,index_col=0,header=0).to_numpy()
X_test_raw = pd.read_csv(X_test_path,index_col=0,header=0).to_numpy()
y_test = pd.read_csv(y_test_path,index_col=0,header=0).to_numpy()
label_names_batch_path = os.path.join(batch_dir,'batches.meta')
labels = preprocess.load_labels(label_names_batch_path)

X_train_norm = preprocess.normalize(X_train_raw)
X_test_norm = preprocess.normalize(X_test_raw)

# filter_red = np.array([[1,0.5,0],[1,1,0],[0,1,1]])
# filter_green = np.array([[0,1,0],[1,1,1],[0,1,0]])
# filter_blue = np.array([[1,0,1],[0,1,0],[1,0,1]])

red_filter = np.array([[-2,-1,-3],[1,0,0],[1,3,3]])
green_filter = np.array([[-1,-2,2],[3,-1,0],[-3,3,2]])
blue_filter = np.array([[3,0,3],[3,-3,0],[3,1,1]])

# red_filter = np.random.randint(low=-3,high=4,size=(3,3))
# green_filter = np.random.randint(low=-3,high=4,size=(3,3))
# blue_filter = np.random.randint(low=-3,high=4,size=(3,3))
channel_filters = np.array([red_filter,green_filter,blue_filter])

bias = -2
print(channel_filters)
print('Applying channel filters and relu function')
X_train = preprocess.get_feature_map_dataset(X_train_norm,channel_filters,bias)
X_test = preprocess.get_feature_map_dataset(X_test_norm,channel_filters,bias)

model_name = "MLP_CLF_convolutionfilter_unoptimized_1"
model_dir = os.path.join(project_dir,model_name)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
print(f'Fitting {model_name}')
clf = MLPCLF(train_dataset=(X_train,y_train),model_dir=model_dir,model_name=model_name)
clf.fit(optimize=False)

print(f'Generating predictions with {model_name}')
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
evaluate.get_metrics(y_true=y_train,y_pred=y_train_pred,labels=labels,train=True,model_dir=model_dir)
evaluate.get_metrics(y_true=y_test,y_pred=y_test_pred,labels=labels,train=False,model_dir=model_dir)

# for i in range(99,149):
#     image = preprocess.normalize(X_train_raw[i])
#     # Reshape to get each channel as 32*32, then move axes for compatibility with imshow
#     image_reshaped = np.moveaxis(image.reshape((3,32,32)),0,-1)
#     image_grayscale = preprocess.rgb_to_grayscale(image)
#     image_channels = preprocess.separate_and_reshape_channels(image)
#     image_red = image_channels[:,:,0]
#     image_green = image_channels[:,:,1]
#     image_blue = image_channels[:,:,2]

#     feature_map = preprocess.get_feature_map(image_channels,channel_filters,bias)

#     print(y_train[i,0])
#     plt.subplot(2,3,1)
#     plt.imshow(image_reshaped)
#     plt.axis('off')
#     plt.title('Original image')
#     plt.subplot(2,3,2)
#     plt.imshow(image_red,cmap='Reds')
#     plt.axis('off')
#     plt.title('Red values')
#     plt.subplot(2,3,3)
#     plt.imshow(image_green,cmap='Greens')
#     plt.axis('off')
#     plt.title('Green values')
#     plt.subplot(2,3,4)
#     plt.imshow(image_blue,cmap='Blues')
#     plt.axis('off')
#     plt.title('Blue values')
#     plt.subplot(2,3,5)
#     plt.imshow(image_grayscale.reshape((32,32)),cmap='binary')
#     plt.axis('off')
#     plt.title('Grayscale values')
#     plt.subplot(2,3,6)
#     plt.imshow(feature_map.reshape((30,30)),cmap='binary')
#     plt.axis('off')
#     plt.title('Conv. filter + ReLU')
#     plt.show()
#     plt.close()
