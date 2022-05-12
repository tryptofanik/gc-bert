import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid, aggr='add')
        self.gc2 = GCNConv(nhid, nclass, aggr='add')
        self.dropout = dropout

    def forward(self, x, edge_idx):
        x = self.gc1(x, edge_idx)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_idx)
        return F.log_softmax(x, dim=1)


class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN2, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid, aggr='add')
        self.gc2 = GCNConv(nhid, nhid, aggr='add')
        self.linear = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_idx):
        x = self.gc1(x, edge_idx)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_idx)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
