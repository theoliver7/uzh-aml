from grakel.utils import graph_from_networkx
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from grakel.kernels import ShortestPath
from grakel import GraphKernel, datasets
import pickle
#DD_data = datasets.fetch_dataset("DD", verbose=False)
#DD_d = DD_data.data
#DD_t = DD_data.target

with open(r"models\sp_kernel_baseline\graphs.pickle", "rb") as f:
    graphs = pickle.load(f)
with open(r"models\sp_kernel_baseline\labels.pickle", "rb") as f:
    labels = pickle.load(f)
   
graphs_g = list(graph_from_networkx(list(graphs.values())))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(graphs_g, labels, test_size=0.2, random_state=42)

# Define kernel
kernel = ShortestPath(normalize=True,with_labels=False)

# Compute kernel matrices
K_train = kernel.fit_transform(X_train)
K_test = kernel.transform(X_test)

# Train SVM with the computed kernel
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)

# Predictions
y_pred = svm.predict(K_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")