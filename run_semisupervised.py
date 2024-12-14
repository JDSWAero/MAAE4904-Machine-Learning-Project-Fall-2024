# Import necessary libraries
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
import os

# Create folder for datasets, model files, results
project_dir = r"C:\Users\jackw\Documents\MAAE4904\Project"

# Load the augmented dataset
with open(os.path.join(project_dir,'train_augmented.pkl'), 'rb') as f:
    data = pickle.load(f)

# Extract augmented data
x_train_augmented = data['x_train_augmented']
y_train_augmented = data['y_train_augmented']

# Separate original and augmented datasets
x_train_original = x_train_augmented[:50000]
y_train_original = y_train_augmented[:50000]

# Normalize pixel values and flatten images for PCA
x_train_original_flat = x_train_original.reshape(50000, -1) / 255.0
x_train_augmented_flat = x_train_augmented.reshape(55000, -1) / 255.0

# Select 10 labeled samples per class for mapping clusters to labels
selected_indices = []
for class_label in range(10):  # CIFAR-10 has 10 classes
    class_indices = np.where(y_train_original == class_label)[0]
    selected_indices.extend(np.random.choice(class_indices, 10, replace=False))

x_labeled = x_train_original_flat[selected_indices]
y_labeled = y_train_original[selected_indices]
x_unlabeled = np.delete(x_train_original_flat, selected_indices, axis=0)
y_unlabeled_true = np.delete(y_train_original, selected_indices, axis=0)

# Reduce dimensionality with PCA to 50 components
pca = PCA(n_components=50, random_state=42)
x_unlabeled_reduced = pca.fit_transform(x_unlabeled)

# Perform K-Means clustering on the original dataset
kmeans = KMeans(
    n_clusters=3000, 
    random_state=42, 
    n_init=20, 
    init='k-means++'
)
y_clustered = kmeans.fit_predict(x_unlabeled_reduced)

# Map clusters to class labels
cluster_label_mapping = {}
for cluster in range(3000):
    cluster_indices = np.where(y_clustered == cluster)[0]
    true_labels = y_unlabeled_true[cluster_indices]
    if len(true_labels) > 0:  # Avoid empty clusters
        most_common_label = np.bincount(true_labels.flatten()).argmax()
        cluster_label_mapping[cluster] = most_common_label

# Assign predicted labels using the cluster mapping
y_predicted = np.array([cluster_label_mapping[cluster] for cluster in y_clustered])

# Evaluate clustering accuracy and generate a confusion matrix
accuracy = accuracy_score(y_unlabeled_true.flatten(), y_predicted)
print(f"Clustering accuracy for original dataset: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_unlabeled_true.flatten(), y_predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(10))
disp.plot(cmap="viridis", xticks_rotation="vertical")
plt.title("Confusion Matrix for Clustering (Original Dataset)")
plt.show()

# Save clustering results for the original dataset
unsupervised_original_data = {
    "x_data": x_unlabeled,
    "y_labels": y_predicted
}
with open("unsupervised_original_data.pkl", "wb") as f:
    pickle.dump(unsupervised_original_data, f)

# Repeat the same process for the augmented dataset
x_augmented_reduced = pca.fit_transform(x_train_augmented_flat)

kmeans_augmented = KMeans(
    n_clusters=3000, 
    random_state=42, 
    n_init=20, 
    init='k-means++'
)
y_clustered_augmented = kmeans_augmented.fit_predict(x_augmented_reduced)

# Map clusters to class labels for the augmented dataset
cluster_label_mapping_augmented = {}
for cluster in range(3000):
    cluster_indices = np.where(y_clustered_augmented == cluster)[0]
    true_labels = y_train_augmented[cluster_indices]
    if len(true_labels) > 0:
        most_common_label = np.bincount(true_labels.flatten()).argmax()
        cluster_label_mapping_augmented[cluster] = most_common_label

y_predicted_augmented = np.array([cluster_label_mapping_augmented[cluster] for cluster in y_clustered_augmented])

# Evaluate clustering accuracy for the augmented dataset
accuracy_augmented = accuracy_score(y_train_augmented.flatten(), y_predicted_augmented)
print(f"Clustering accuracy for augmented dataset: {accuracy_augmented * 100:.2f}%")

conf_matrix_augmented = confusion_matrix(y_train_augmented.flatten(), y_predicted_augmented)
disp_augmented = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_augmented, display_labels=range(10))
disp_augmented.plot(cmap="viridis", xticks_rotation="vertical")
plt.title("Confusion Matrix for Clustering (Augmented Dataset)")
plt.show()

# Save clustering results for the augmented dataset
unsupervised_augmented_data = {
    "x_data": x_train_augmented,
    "y_labels": y_predicted_augmented
}
with open("unsupervised_augmented_data.pkl", "wb") as f:
    pickle.dump(unsupervised_augmented_data, f)