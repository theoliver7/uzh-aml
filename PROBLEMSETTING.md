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
1. We start by rebuilding the proposed model from the original paper 
2. The next step is to apply the following modifications and check them for possible improvements:
    1. Modify the read out function (don't use all the pooled graphs for readout)
    2. Introduce a deeper network with an eventual modification of the convolution layer such that the downsampling gets reduced.
    3. Use a different node information score metric (e.g., instead of using the Manhatten distance, we use the Euclidean distance)
    4. Different top-rank node selection (based on node information score)

## Evaluation
- Be sure to define the evaluation pipeline (state the train/test split, make sure to work with same data in order to ensure comparability, WRITE MORE STUFF, define a concrete baseline)
- The evaluation metrics are going to be the same as in the _Hierarchical Graph Pooling with Structure Learning_ paper which means that we focus on accuracy. This way we ensure equal test settings. 
To make our numbers comparable to the ones from the paper we follow the same evaluation protocol
```
1. Randomly split each dataset into three parts: 80% as training set, 10% as validation set and the remaining 10% as test set. 
2. We repeat this randomly splitting process 10 times
3. Report the average performance with standard derivation
```
### Baselines
#### Statistical Model: Graphlets
As the original paper we can compare our results to a statistical baseline like Graphlets (http://proceedings.mlr.press/v5/shervashidze09a.html)
#### SOTA: 
According to papers with code the SOTA on this Dataset is MEWISPool, that are trained in an supervised fashion (https://paperswithcode.com/sota/graph-classification-on-dd) 

