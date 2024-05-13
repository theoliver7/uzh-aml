from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from grakel.kernels import ShortestPath

from GraKeL.grakel.datasets.base import fetch_dataset

DD_data = fetch_dataset("DD", verbose=False)
DD_d = DD_data.data
DD_t = DD_data.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(DD_d, DD_t, test_size=0.1, random_state=777)

# Define kernel
kernel = ShortestPath(normalize=True, with_labels=False)

# Compute kernel matrices
K_train = kernel.fit_transform(X_train)
K_test = kernel.transform(X_test)

# Train SVM with the computed kernel
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)

y_train_pred = svm.predict(K_train)
accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {accuracy:.4f}")

# Predictions
y_pred = svm.predict(K_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")