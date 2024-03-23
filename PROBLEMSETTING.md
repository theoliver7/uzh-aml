# Project proposal
Carlos Kirchdorfer, carlos.kirchdorfer@uzh.ch, 19-720-002 \
Oliver Aschwanden, oliverrobin.aschwanden@uzh.ch, 19-874-627 \
Marco Heiniger, marco.heiniger@uzh.ch, 18-733-824

## Overview
The goal of this project is to explore further possibilities of improving the approach of the paper _Hierarchical Graph Pooling with Structure Learning_ (https://arxiv.org/abs/1911.05954). In general, the paper tries to improve predictions with Graph Neural Networks (GNN). In more detail, they have introduced a new approach to graph pooling, wherein they adaptively select a subset of nodes to form an induced subgraph for the subsequent layer. This selection is accompanied by a structure learning mechanism that refines the graph structure for the pooled graph at each layer. The aim is to preserve the integrity of the graph's topological information.


## Problem formulation
In the field of bioinformatics, protein structures can be modeled as a graph, where the nodes have the meaning of amino acides. Two nodes have an edge between them if they are less than 6 Angstroms apart. The [D&D Dataset](https://pubmed.ncbi.nlm.nih.gov/12850146/) contains a set of 1178 of these proteins graphs $\mathcal{G} = \{G_1, G_2, \dots, G_n\}$ where in this case, $n$ = 1178. For an arbitrary protein graph $G_i = (V_i, E_i, X_i)$, we have $n_i$ and $e_i$ that denote the number of amino acids (nodes) and connections (edges), respectively. Let for one protein graph $G_i$, let $\mathbf{X_i} \in \mathbb{R}^{n_i \times f}$ represent the node feature matrix, where $f$ is the dimension of the node attributes and serves to characterize the amino acid. The protein graph can then either be classified to be an enzyme or not. The target labels for the 1178 different protein graphs can be represented through a graph label vector $\mathbf{y} \in \mathbb{R}^{n}$. Now we can define our problem the following way:

**Training:** We train our graph classification model with a set of protein graphs $\mathcal{G}_L$ whose label information $\mathbf{y}_L$ is given in a vector form (more about the splitting in the Evaluation section). The label information vector serves as the target vector which is needed in order to train the network in a supervised way.

**Task and Goal:** The classification task (output) is to decide, given some protein graph as input, if a protein structure $G_i$ should be classified as an enzym $y_{i} = 1$ , or not $y_{i} = 0$. The goal is that through learning the amino acide (node) features and their connections (edges) in the training phase, the network improves in classifying the proteins correctly. Importantly, we aim to achieve good results also on unseen graphs (graphs which have not been included in the training set), as the performance on these graphs indicate how well the network generalizes.

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
__or maybe something from here: https://github.com/jajupmochi/graphkit-learn?tab=readme-ov-file__


