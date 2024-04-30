import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv

from models.hgp_sl_py311.layers_new import GCN, HGPSLPool


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args["num_features"]
        self.nhid = args["nhid"]
        self.num_classes = args["num_classes"]
        self.pooling_ratio = args["pooling_ratio"]
        self.dropout_ratio = args["dropout_ratio"]
        self.negative_slope = args["negative_slope"]
        self.sample = args["sample_neighbor"]
        self.sparse = args["sparse_attention"]
        self.sl = args["structure_learning"]
        self.lamb = args["lamb"]

        self.dist = args["dist"]

        self.convolutions = torch.nn.ModuleList()

        self.convolutions.append(GCNConv(self.num_features, self.nhid))
        self.pooling_layers = torch.nn.ModuleList()
        for _ in range(args['num_layers']-1):
            self.convolutions.append(GCN(self.nhid, self.nhid))

            self.pooling_layers.append(
                HGPSLPool(self.nhid,
                          self.pooling_ratio,
                          self.sample,
                          self.sparse,
                          self.sl,
                          self.lamb,
                          self.negative_slope,
                          self.dist))

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_attr = None
        concat = []
        for idx, conv in enumerate(self.convolutions):
            x = F.relu(conv(x, edge_index, edge_attr))
            if idx != len(self.convolutions) - 1:
                x, edge_index, edge_attr, batch = self.pooling_layers[idx](x, edge_index, edge_attr, batch)
            concat.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))

        x = sum(F.relu(x_layer) for x_layer in concat)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
