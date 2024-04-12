import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gklearn.kernels import spkernel 
from sklearn.metrics import accuracy_score
import pickle
import multiprocessing
import networkx as nx
def load():
    with open("graphs.pickle", "rb") as f:
        graphs = pickle.load(f)
    with open("labels.pickle", "rb") as f:
        labels = pickle.load(f)
    return graphs, labels



class GK_SP:
    """
    Shorthest path graph kernel.
    """
    def compare(self, g_1, g_2, verbose=False):
        """Compute the kernel value (similarity) between two graphs.

        Parameters
        ----------
        g1 : networkx.Graph
            First graph.
        g2 : networkx.Graph
            Second graph.

        Returns
        -------
        k : The similarity value between g1 and g2.
        """
        # Diagonal superior matrix of the floyd warshall shortest
        # paths:
        fwm1 = np.array(nx.floyd_warshall_numpy(g_1))
        fwm1 = np.where(fwm1 == np.inf, 0, fwm1)
        fwm1 = np.where(fwm1 == np.nan, 0, fwm1)
        fwm1 = np.triu(fwm1, k=1)
        bc1 = np.bincount(fwm1.reshape(-1).astype(int))

        fwm2 = np.array(nx.floyd_warshall_numpy(g_2))
        fwm2 = np.where(fwm2 == np.inf, 0, fwm2)
        fwm2 = np.where(fwm2 == np.nan, 0, fwm2)
        fwm2 = np.triu(fwm2, k=1)
        bc2 = np.bincount(fwm2.reshape(-1).astype(int))

        # Copy into arrays with the same length the non-zero shortests
        # paths:
        v1 = np.zeros(max(len(bc1), len(bc2)) - 1)
        v1[range(0, len(bc1)-1)] = bc1[1:]

        v2 = np.zeros(max(len(bc1), len(bc2)) - 1)
        v2[range(0, len(bc2)-1)] = bc2[1:]

        return np.sum(v1 * v2)

    def compare_normalized(self, g_1, g_2, verbose=False):
        """Compute the normalized kernel value between two graphs.

        A normalized version of the kernel is given by the equation:
        k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2))

        Parameters
        ----------
        g1 : networkx.Graph
            First graph.
        g2 : networkx.Graph
            Second graph.

        Returns
        -------
        k : The similarity value between g1 and g2.

        """
        return self.compare(g_1, g_2) / (np.sqrt(self.compare(g_1, g_1) *
                                                 self.compare(g_2, g_2)))

    def compare_list2(self, graph_list,graph_list2, verbose=False):
        """Compute the all-pairs kernel values for a list of graphs.

        This function can be used to directly compute the kernel
        matrix for a list of graphs. The direct computation of the
        kernel matrix is faster than the computation of all individual
        pairwise kernel values.

        Parameters
        ----------
        graph_list: list
            A list of graphs (list of networkx graphs)

        Return
        ------
        K: numpy.array, shape = (len(graph_list), len(graph_list))
        The similarity matrix of all graphs in graph_list.

        """
        n = len(graph_list)
        m = len(graph_list2)
        k = np.zeros((n, m))
        for i in range(n):
            for j in range(i, m):
                k[i, j] = self.compare(graph_list[i], graph_list2[j])
                k[j, i] = k[i, j]

        k_norm = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])

        return k_norm
  

    def compare_list(self, graph_list1, graph_list2, verbose=False):
        """
        angepasste funktion
        """
        n = len(graph_list1)
        m = len(graph_list2)
        k = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                k[i, j] = self.compare(graph_list1[i], graph_list2[j])

        k_norm = np.zeros(k.shape)
        for i in range(min(n, m)):
            for j in range(min(n, m)):
                k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])

        return k_norm


if __name__ == '__main__':
    graphs,labels= load()
    graphs =list(graphs.values())
    graphs = graphs[680:700]
    labels = labels[680:700]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(graphs, labels, test_size=0.2, random_state=42)
    
    # Compute shortest path kernels using graphkit-learn
    shortest_path = GK_SP()

    #shortest_path_kernel_train, _ , _ = spkernel(X_train,n_jobs=1) # funktioniert nicht mit spkernel
    #shortest_path_kernel_test,_ , _ = spkernel(X_test, X_train,n_jobs=1) 
    shortest_path_kernel_train = shortest_path.compare_list(X_train,X_train)
    shortest_path_kernel_test = shortest_path.compare_list(X_test,X_train)
  
    print("created kernel")

    # Create and train an SVM classifier
    clf = SVC(kernel='precomputed')  # Use precomputed kernel
    clf.fit(shortest_path_kernel_train, y_train)
    print("finished training")
   
    y_pred = clf.predict(shortest_path_kernel_test)
    print("finished predicting")
   
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
