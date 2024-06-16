import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def mask(g, mask_rate=0.5):
    num_nodes = g.num_nodes()
    perm = torch.randperm(num_nodes, device=g.device)
    num_mask_nodes = int(mask_rate * num_nodes)

    mask_nodes = perm[: num_mask_nodes]

    return mask_nodes


class Encoder1(nn.Module):
    def __init__(self, in_hidden, out_hidden, n):
        super(Encoder1, self).__init__()
        self.n = n
        self.conv1 = GraphConv(in_hidden, out_hidden, norm='both',
                               bias=True, activation=nn.PReLU())
        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.actions = nn.ModuleList()

        self.layers.append(self.conv1)
        self.bn.append(BatchNorm(out_hidden))
        self.actions.append(nn.PReLU())

        for _ in range(1, self.n):
            self.layers.append(GraphConv(out_hidden, out_hidden, norm='both',
                               bias=True, activation=nn.PReLU()))
            self.bn.append(BatchNorm(out_hidden))
            self.actions.append(nn.PReLU())

        self.pooling = SumPooling()

    def forward(self, graph, heat):

        h = self.conv1(graph, heat)
        h = self.bn[0](h)
        h = self.actions[0](h)
        gh = self.pooling(graph, h)

        for i in range(1, self.n):
            h = self.layers[i](graph, h)
            h = self.bn[i](h)
            h = self.actions[i](h)
            #gh = torch.cat((gh, self.pooling(graph, h)), -1)
            gh += self.pooling(graph, h)

        return h, gh


class Encoder2(nn.Module):
    def __init__(self, in_hidden, out_hidden, n):
        super(Encoder2, self).__init__()
        self.n = n
        self.conv1 = GraphConv(in_hidden, out_hidden, norm='none',
                               bias=True, activation=nn.PReLU())
        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.actions = nn.ModuleList()

        self.bn.append(BatchNorm(out_hidden))
        self.layers.append(self.conv1)
        self.actions.append(nn.PReLU())

        for _ in range(1, self.n):
            self.layers.append(GraphConv(out_hidden, out_hidden, norm='none',
                                         bias=True, activation=nn.PReLU()))
            self.bn.append(BatchNorm(out_hidden))
            self.actions.append(nn.PReLU())
        self.pooling = SumPooling()

    def forward(self, diff_graph, heat):

        h = self.conv1(diff_graph, heat, edge_weight=diff_graph.edata['diff_weight'])
        h = self.bn[0](h)
        h = self.actions[0](h)
        gh = self.pooling(diff_graph, h)

        for i in range(1, self.n):
            h = self.layers[i](diff_graph, h, edge_weight=diff_graph.edata['diff_weight'])
            h = self.bn[i](h)
            h = self.actions[i](h)
            gh += self.pooling(diff_graph, h)
            #gh = torch.cat((gh, self.pooling(diff_graph, h)), -1)

        return h, gh


class MG(nn.Module):
    def __init__(self, in_hidden, out_hidden, rate, rate1, alpha, n1, n2):
        super(MG, self).__init__()
        self.enc = Encoder1(in_hidden, out_hidden, n1)
        self.dec = Encoder2(in_hidden, out_hidden, n2)
        self.rate = rate
        self.rate1 = rate1
        self.alpha = alpha
        self.n1 = n1
        self.n2 = n2

    def forward(self, graph, diff_graph):

        mask_nodes = mask(graph, mask_rate=self.rate)
        x = graph.ndata["attr"].clone()
        x[mask_nodes] = 0.0

        h1, gh1 = self.enc(graph, x)
        h2, gh2 = self.dec(diff_graph, graph.ndata["attr"])
        loss1 = sce_loss(h1[mask_nodes], h2[mask_nodes])

        mask_nodes1 = mask(graph, mask_rate=self.rate1)
        x = graph.ndata["attr"].clone()
        x[mask_nodes1] = 0.0

        h1, gh1 = self.enc(graph, graph.ndata["attr"])
        h2, gh2 = self.dec(diff_graph, x)
        loss2 = sce_loss(h1[mask_nodes1], h2[mask_nodes1])

        return self.alpha * loss1 + loss2 * (1 - self.alpha)

    def get_embed(self, graph, diff_graph):
        h1, gh1 = self.enc(graph, graph.ndata['attr'])
        h2, gh2 = self.dec(diff_graph, graph.ndata['attr'])

        return gh1, gh2
