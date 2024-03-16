# Project proposal
Carlos Kirchdorfer, carlos.kirchdorfer@uzh.ch, 19-720-002 \
Oliver Aschwanden, oliverrobin.aschwanden@uzh.ch, 19-874-627 \
Marco Heiniger, marco.heiniger@uzh.ch, 18-733-824

## Overview
The goal of this project is to explore further possibilities of improving the approach of the paper _Hierarchical Graph Pooling with Structure Learning_ (https://arxiv.org/abs/1911.05954). In general, the paper tries to improve predictions with Graph Neural Networks (GNN). In more detail, they have introduced a new approach to graph pooling, wherein they adaptively select a subset of nodes to form an induced subgraph for the subsequent layer. This selection is accompanied by a structure learning mechanism that refines the graph structure for the pooled graph at each layer. The aim is to preserve the integrity of the graph's topological information.


## Problem formulation
In the field of bioinformatic proteins structures can be modeled as a graph, where the nodes represent amino acides, which have an edge if they are less than 6 Angstroms apart. The [D&D Dataset](https://pubmed.ncbi.nlm.nih.gov/12850146/) contains a set 1178 of these proteins graphs $\\mathcal{G} = \\{G_1, G_2, \\dots, G_n\\}$. Each protein within this collection may contain a variable number of amino acids(nodes) and connections(edges). For an arbitrary graph $G_i = (V_i, E_i, X_i)$, we have $n_i$ and $e_i$ denote the number of amino acids and connections, respectively. Let $\mathbf{A_i} \in \mathbb{R}^{n_i \times n_i}$ be the adjacency matrix describing its edge connections and $\mathbf{X_i} \in \mathbb{R}^{n_i \times f}$ represent the node feature matrix, where $f$ is the dimension of node attributes. These protein structure can either be enzymes or non-enzymes represented through a graph label matrix $\mathbf{Y} \in \mathbb{R}^{n \times c}$.Now we can define our problem the following way:

**Input:** We start with a set of graphs $\mathcal{G}_L$ with its label information $\mathbf{Y}_L$, ~~the number of graph neural network layers $K$, pooling ratio $r$, and dimension $d$ in each layer.~~

**Task:** The classification task is to decide if a protein structure $G_i$ belongs to class the enzym $Y_{ij} = 1$ , or is a non-enzym $Y_{ij} = 0$.

## Approach and self-contributions
1. We start by rebuilding the proposed model from the original paper. 
2. Then we will implement the simple baseline model Graphlets 
3. The next step is to apply the following modifications and check them for possible improvements:
    1. Modify the read out function (don't use all the pooled graphs for readout)
    2. Introduce a deeper network with an eventual modification of the convolution layer such that the downsampling gets reduced.
    3. Use a different node information score metric (e.g., instead of using the Manhatten distance, we use the Euclidean distance)
    4. Different top-rank node selection (based on node information score)

## Evaluation
The evaluation protocol is going to be the same as in the _Hierarchical Graph Pooling with Structure Learning_ paper. In this way, we ensure our numbers are comparable to those presented in the paper by adhering to the evaluation protocol outlined as follows:
1. Randomly split each dataset into three parts: 80% as training set, 10% as validation set and the remaining 10% as test set. 
2. We repeat this randomly splitting process 10 times
3. Report the average performance with standard derivation

### Baselines
- **HPG-SL**:
Our primary goal will be to surpass the paper's baseline, which was 80.96% accuracy
- **Graphlets:**
To have a simple statistical baseline we will use the graph kernel model Graphlets. We will mainly stick to the original paper for the implementation. (http://proceedings.mlr.press/v5/shervashidze09a.html)
- **MEWISPool:** 
We also acknowledge the current SOTA model (according to paperswithcode.com) as comparison which was tested on the D&D dataset with an accuracy of 84.33%.(https://paperswithcode.com/sota/graph-classification-on-dd) 


