from grakel.kernels import ShortestPath
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from GraKeL.grakel.datasets.base import fetch_dataset

DD_data = fetch_dataset("DD", verbose=False)
DD_d = DD_data.data
DD_t = DD_data.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(DD_d, DD_t, test_size=0.1, random_state=42)
# Define kernel
kernel = ShortestPath(normalize=True, with_labels=False)

# Compute kernel matrices
K_train = kernel.fit_transform(X_train)
K_test = kernel.transform(X_test)

# Train SVM with the computed kernel
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)


kf = KFold(n_splits=5)

accuracy = cross_val_score(svm, K_train, y_train, cv=kf, scoring='accuracy')

print(f"KFold Accuracy: {accuracy.mean():.4f}")
print(accuracy)


# Predictions
y_pred = svm.predict(K_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")