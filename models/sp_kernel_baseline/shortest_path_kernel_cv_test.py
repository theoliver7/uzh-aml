from grakel.kernels import ShortestPath
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from GraKeL.grakel.datasets.base import fetch_dataset

DD_data = fetch_dataset("DD", verbose=False)
DD_d = DD_data.data
DD_t = DD_data.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(DD_d, DD_t, test_size=0.1, random_state=777)
X_train = np.array(X_train)
y_train = np.array(y_train)
# Define kernel
kf = KFold(n_splits=5)
accuracies = []
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val = X_train[train_index], X_train[val_index]
    y_train_fold, y_val = y_train[train_index], y_train[val_index]
    kernel = ShortestPath(normalize=True, with_labels=False)

    # Compute kernel matrices
    K_train = kernel.fit_transform(X_train_fold)
    K_val = kernel.transform(X_val)

    # Train SVM with the computed kernel
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train_fold)

    y_pred = svm.predict(K_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Val Accuracy: {accuracy:.4f}")
    accuracies.append(accuracy)

print(f"KFold Accuracy: {accuracies.mean():.4f}")
print(accuracy)

# Predictions
# y_pred = svm.predict(K_val)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")
