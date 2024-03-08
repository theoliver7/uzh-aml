# Project proposal

Given a set of graph data $\mathcal{G} = \{G_1, G_2, \dots, G_n\}$, where the number of nodes and edges in each graph might be quite different. For an arbitrary graph $G_i = (V_i, E_i, X_i)$, we have $n_i$ and $e_i$ denote the number of nodes and edges, respectively. Let $\mathbf{A}_i \in \mathbb{R}^{n_i \times n_i}$ be the adjacency matrix describing its edge connection information and $\mathbf{X}_i \in \mathbb{R}^{n_i \times f}$ represent the node feature matrix, where $f$ is the dimension of node attributes. Label matrix $\mathbf{Y} \in \mathbb{R}^{n \times c}$ indicates the associated labels for each graph, i.e., if $G_i$ belongs to class $j$, then $Y_{ij} = 1$, otherwise $Y_{ij} = 0$. Since the graph structure and node numbers change between layers due to the graph pooling operation, we further represent the $i$-th graph fed into the $k$-th layer as $G_k^i$ with $n_k^i$ nodes. The adjacency matrix and hidden representation matrix are then denoted as $\mathbf{A}_k^i \in \mathbb{R}^{n_k^i \times n_k^i}$ and $\mathbf{H}_k^i \in \mathbb{R}^{n_k^i \times d}$. With the above notations, we formally define our problem as follows:

**Input:** Given a set of graphs $\mathcal{G}_L$ with its label information $\mathbf{Y}_L$, the number of graph neural network layers $K$, pooling ratio $r$, and representation dimension $d$ in each layer.

**Output:** Our goal is to predict the unknown graph labels of $\mathcal{G}/\mathcal{G}_L$ with graph neural network in an end-to-end way.

