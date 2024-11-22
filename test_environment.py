import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

import preprocess

project_dir = r"C:\Users\jackw\Documents\MAAE4904\Project"
X_train_path = os.path.join(project_dir,'merged_X_train_batches.csv')
y_train_path = os.path.join(project_dir,'merged_y_train_batches.csv')
X_train = read_csv(X_train_path,index_col=0,header=0).to_numpy()
y_train = read_csv(y_train_path,index_col=0,header=0).to_numpy()

i = 19999
image = X_train[i]
# Reshape to get each channel as 32*32, then move axes for compatibility with imshow
image_reshaped = np.moveaxis(image.reshape((3,32,32)),0,-1)
image_grayscale = preprocess.rgb_to_grayscale(image)

print(y_train[i,0])
plt.subplot(1,2,1)
plt.imshow(image_reshaped)
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(image_grayscale.reshape((32,32)),cmap='binary')
plt.axis('off')

plt.show()
plt.close()
