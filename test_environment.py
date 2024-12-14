import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

import preprocess

project_dir = r"C:\Users\jackw\Documents\MAAE4904\Project"
X_train_path = os.path.join(project_dir,'merged_X_train_batches.csv')
y_train_path = os.path.join(project_dir,'merged_y_train_batches.csv')
# X_test_path = os.path.join(project_dir,'X_test_batch.csv')
# y_test_path = os.path.join(project_dir,'y_test_batch.csv')
X_train = read_csv(X_train_path,index_col=0,header=0).to_numpy()
y_train = read_csv(y_train_path,index_col=0,header=0).to_numpy()
# X_test = read_csv(X_test_path,index_col=0,header=0).to_numpy()
# y_test = read_csv(y_test_path,index_col=0,header=0).to_numpy()

X_train_norm = preprocess.normalize(X_train)
# X_test_norm = preprocess.normalize(X_test)

i = 499
image = X_train[i]
# Reshape to get each channel as 32*32, then move axes for compatibility with imshow
image_reshaped = np.moveaxis(image.reshape((3,32,32)),0,-1)
image_grayscale = preprocess.rgb_to_grayscale(image)
image_red, image_green, image_blue = preprocess.separate_and_reshape_channels(image)

print(y_train[i,0])
plt.subplot(2,3,1)
plt.imshow(image_reshaped)
plt.axis('off')
plt.subplot(2,3,2)
plt.imshow(image_red,cmap='Reds')
plt.axis('off')
plt.subplot(2,3,3)
plt.imshow(image_green,cmap='Greens')
plt.axis('off')
plt.subplot(2,3,4)
plt.imshow(image_blue,cmap='Blues')
plt.axis('off')
plt.subplot(2,3,5)
plt.imshow(image_grayscale.reshape((32,32)),cmap='binary')
plt.axis('off')
plt.show()
plt.close()

# batch_dir = os.path.join(project_dir,r"cifar-10-python\cifar-10-batches-py")
# test_batch_path = os.path.join(batch_dir,'test_batch')
# label_names_batch_path = os.path.join(batch_dir,'batches.meta')
# X_test, y_test = preprocess.load_x_y(test_batch_path)
# labels = preprocess.load_labels(label_names_batch_path)
# DataFrame(X_test).to_csv(os.path.join(project_dir,'X_test_batch.csv'))
# DataFrame(y_test).to_csv(os.path.join(project_dir,'y_test_batch.csv'))
# DataFrame(labels).to_csv(os.path.join(project_dir,'labels.csv'))

# print('Performing PCA')
# pca_dir = os.path.join(project_dir,'PCA')
# pca = preprocess.get_PCA(X_train=X_train_norm,n_components=0,pca_dir=pca_dir)
# print(pca.explained_variance_)
# plt.plot(pca.explained_variance_)
# plt.show()
# plt.close()
# X_train_norm_transformed = preprocess.apply_PCA(pca_n=pca,X=X_train_norm)
# X_test_norm_transformed = preprocess.apply_PCA(pca_n=pca,X=X_test_norm)
