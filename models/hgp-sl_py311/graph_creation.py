import pickle
import numpy as np
import networkx as nx
def create_networkx_graphs(data_folder, dataset_name):
  # Create a dictionary to store nodes for each graph
  graph_nodes = {}
  with open(f"{data_folder}/{dataset_name}_graph_indicator.txt", "r") as f:
    for i, line in enumerate(f):
      graph_id = int(line.strip())
      if graph_id not in graph_nodes:
        graph_nodes[graph_id] = []
      graph_nodes[graph_id].append(i)

  # Create the NetworkX graphs
  graphs = {}
  with open(f"{data_folder}\{dataset_name}_A.txt", "r") as f:
    for line in f:
      source, target = map(int, line.strip().split(","))
      # Check which graph this edge belongs to
      for graph_id, nodes in graph_nodes.items():
        if source in nodes and target in nodes:
          if graph_id not in graphs:
            graphs[graph_id] = nx.Graph() 
          graphs[graph_id].add_edge(source, target)
  
  with open("graphs.pickle", "wb") as f:
      pickle.dump(graphs, f)
  

def load_labels(data_folder,filename):
  with open(f"{data_folder}\{filename}_graph_labels.txt", "r") as f:
    lines = f.readlines()
    labels = np.array([int(line.strip()) for line in lines], dtype=int)
  with open("labels.pickle", "wb") as f:
      pickle.dump(labels, f)

if __name__ == '__main__':
    data_folder = "data\DD"
    dataset_name = "DD"
    print("started networkx graph creation")
    #graphs_dict = create_networkx_graphs(data_folder, dataset_name)
    labels = load_labels(data_folder,dataset_name)
    print("finished preprocessing")