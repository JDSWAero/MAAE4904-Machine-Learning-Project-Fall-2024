import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

import preprocess

project_dir = r"C:\Users\jackw\Documents\MAAE4904\Project"
X_train_path = os.path.join(project_dir,'merged_X_train_batches.csv')
X_train = read_csv(X_train_path,index_col=0,header=0).to_numpy()

image = X_train[1999]
print(image.reshape((32,32,3)))
image_grayscale = preprocess.rgb_to_grayscale(image)
print(image_grayscale)

plt.imshow(image.reshape((32,32,3)),interpolation='none')
plt.axis('off')

# plt.imshow(image_grayscale.reshape((32,32)),cmap='binary')
# plt.axis('off')

plt.show()
plt.close()
