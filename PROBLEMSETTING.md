# Project proposal
Carlos Kirchdorfer, carlos.kirchdorfer@uzh.ch, 19-720-002 \
Oliver Aschwanden, oliverrobin.aschwanden@uzh.ch, 19-874-627 \
Marco Heiniger, marco.heiniger@uzh.ch, 18-733-824

## Overview
The goal of this project is to explore further possibilities of improving the approach of the paper _Hierarchical Graph Pooling with Structure Learning_ (https://arxiv.org/abs/1911.05954). In general, the paper tries to improve predictions with Graph Neural Networks (GNN). In more detail, they have introduced a new approach to graph pooling, wherein they adaptively select a subset of nodes to form an induced subgraph for the subsequent layer. This selection is accompanied by a structure learning mechanism that refines the graph structure for the pooled graph at each layer. The aim is to preserve the integrity of the graph's topological information.


## Problem formulation
Consider a set of graphs $\\mathcal{G} = \\{G_1, G_2, \\dots, G_n\\}$. Each graph within this collection may contain a variable number of nodes and edges. For an arbitrary graph $G_i = (V_i, E_i, X_i)$, we have $n_i$ and $e_i$ denote the number of nodes and edges, respectively. Let $\mathbf{A_i} \in \mathbb{R}^{n_i \times n_i}$ be the adjacency matrix describing its edge connections and $\mathbf{X_i} \in \mathbb{R}^{n_i \times f}$ represent the node feature matrix, where $f$ is the dimension of node attributes. Label matrix $\mathbf{Y} \in \mathbb{R}^{n \times c}$ indicates the associated labels for each graph. For example if $G_i$ belongs to class $j$, then $Y_{ij} = 1$, else $Y_{ij} = 0$. Now we can define our problem the following way:

**Input:** We start with a set of graphs $\mathcal{G}_L$ with its label information $\mathbf{Y}_L$, the number of graph neural network layers $K$, pooling ratio $r$, and dimension $d$ in each layer.

**Output:** The aim of the task is to predict the unknown graph labels of $\mathcal{G}/\mathcal{G}_L$ with our graph neural network implementation.

- More concrete, focus on data set, (Consider a moluecule, which are a set of graphs and predict if they cause cancer, nodes == atoms, ), we focus on D&D data set (there are also different papers on paperswithcode from which we can get "inspiration".)

## Approach and self-contributions
1. We start by rebuilding the proposed model from the original paper 
2. The next step is to apply the following modifications and check them for possible improvements:
    1. Modify the read out function (don't use all the pooled graphs for readout)
    2. Introduce a deeper network with an eventual modification of the convolution layer such that the downsampling gets reduced.
    3. Use a different node information score metric (e.g., instead of using the Manhatten distance, we use the Euclidean distance)
    4. Different top-rank node selection (based on node information score)


Importantly, it should be noted that we are solely focusing on one dataset out of those tested in the paper, and further modifications, not listed, may be implemented during the project

## Evaluation
- Be sure to define the evaluation pipeline (state the train/test split, make sure to work with same data in order to ensure comparability, WRITE MORE STUFF, define a concrete baseline)

- Baseline we should include: Very simple ML/statistical model, state of the art ML baseline
-> Be more concrete about baseline formulation
  
The evaluation metrics are going to be the same as in the _Hierarchical Graph Pooling with Structure Learning_ paper which means that we focus on accuracy. This way we ensure equal test settings. 

Moreover we will also implement a basic GNN based on the paper _A Simple Baseline Algorithm for Graph Classification_ (https://arxiv.org/pdf/1810.09155.pdf) such that we can compare it to our more complex model.
