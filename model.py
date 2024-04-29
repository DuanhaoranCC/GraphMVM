import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def mask(g, x, mask_rate=0.5):
    num_nodes = g.num_nodes()
    perm = torch.randperm(num_nodes, device=x.device)
    num_mask_nodes = int(mask_rate * num_nodes)

    # random masking
    # num_mask_nodes = int(mask_rate * num_nodes)
    mask_nodes = perm[: num_mask_nodes]

    return mask_nodes


class Encoder1(nn.Module):
    def __init__(self, in_hidden, out_hidden, p, n):
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
        self.dp = nn.Dropout(p)

    def forward(self, graph, diff_graph, heat, edge_weight):
        x = self.dp(heat)
        h = self.conv1(graph, x)
        h = self.bn[0](h)
        h = self.actions[0](h)

        for i in range(1, self.n):
            h = self.layers[i](graph, h)
            h = self.bn[i](h)
            h = self.actions[i](h)


        return h



class Encoder2(nn.Module):
    def __init__(self, in_hidden, out_hidden, p, n):
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
        self.dp = nn.Dropout(p)

    def forward(self, graph, diff_graph, heat, edge_weight):
        x = self.dp(heat)
        h = self.conv1(diff_graph, x, edge_weight=edge_weight)
        h = self.bn[0](h)
        h = self.actions[0](h)

        for i in range(1, self.n):
            h = self.layers[i](diff_graph, h, edge_weight=edge_weight)
            h = self.bn[i](h)
            h = self.actions[i](h)

        return h


class MG(nn.Module):
    def __init__(self, in_hidden, out_hidden, p1, p2, rate,
                 rate1, alpha, n1, n2):
        super(MG, self).__init__()
        self.enc = Encoder1(in_hidden, out_hidden, p1, n1)
        self.dec = Encoder1(in_hidden, out_hidden, p2, n2)
        self.rate = rate
        self.rate1 = rate1
        self.alpha = alpha
        # self.noise = 0.05
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_hidden))
        self.enc_mask_token1 = nn.Parameter(torch.zeros(1, in_hidden))
        # self.encoder_to_decoder = nn.Linear(out_hidden, out_hidden, bias=False)
        self.criterion = self.setup_loss_fn("sce")

    def setup_loss_fn(self, loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, graph, diff_graph, feat, edge_weight):

        mask_nodes = mask(graph, feat, mask_rate=self.rate)
        x = feat.clone()
        x[mask_nodes] = 0.0
        x[mask_nodes] += self.enc_mask_token

        h1 = self.enc(graph, diff_graph, x, edge_weight)
        h2 = self.dec(graph, diff_graph, feat, edge_weight)
        loss1 = self.criterion(h1[mask_nodes], h2[mask_nodes])

        mask_nodes1 = mask(graph, feat, mask_rate=self.rate1)
        x = feat.clone()
        x[mask_nodes1] = 0.0
        x[mask_nodes1] += self.enc_mask_token1

        h1 = self.enc(graph, diff_graph, feat, edge_weight)
        h2 = self.dec(graph, diff_graph, x, edge_weight)
        loss2 = self.criterion(h1[mask_nodes1], h2[mask_nodes1])
        # return loss2
        return self.alpha * loss1 + loss2 * (1 - self.alpha)

    def get_embed(self, graph, diff_graph, feat, edge_weight):
        h1 = self.enc(graph, diff_graph, feat, edge_weight)
        h2 = self.dec(graph, diff_graph, feat, edge_weight)

        return h1, h2


# class MG(nn.Module):
#     def __init__(self, in_hidden, out_hidden, p1, p2, beta, beta1, rate,
#                  rate1, alpha, n, n1):
#         super(MG, self).__init__()
#         self.enc = Encoder1(in_hidden, out_hidden, p1)
#         self.dec = Encoder2(in_hidden, out_hidden, p2)
#         self.rate = rate
#         self.rate1 = rate1
#         self.alpha = alpha
#         self.noise = n
#         self.noise1 = n1
#         self.enc_mask_token = nn.Parameter(torch.zeros(1, in_hidden))
#         self.enc_mask_token1 = nn.Parameter(torch.zeros(1, in_hidden))
#         # self.encoder_to_decoder = nn.Linear(out_hidden, out_hidden, bias=False)
#         self.criterion = self.setup_loss_fn("sce", beta)

#     def setup_loss_fn(self, loss_fn, alpha_l):
#         if loss_fn == "mse":
#             criterion = nn.MSELoss()
#         elif loss_fn == "sce":
#             criterion = partial(sce_loss, alpha=alpha_l)
#         else:
#             raise NotImplementedError
#         return criterion

#     def forward(self, graph, diff_graph, feat, edge_weight):

#         token_nodes, noise_nodes, \
#         noise_to_be_chosen, mask_nodes = mask(graph, feat, mask_rate=self.rate,
#                                               noise=self.noise)
#         x = feat.clone()
#         x[token_nodes] = 0.0
#         x[noise_nodes] = feat[noise_to_be_chosen]
#         x[token_nodes] += self.enc_mask_token

#         h1 = self.enc(graph, diff_graph, x, edge_weight)

#         token_nodes1, noise_nodes1, \
#         noise_to_be_chosen1, mask_nodes1 = mask(graph, feat, mask_rate=self.rate1,
#                                                 noise=self.noise1)
#         x = feat.clone()
#         x[token_nodes1] = 0.0
#         x[noise_nodes1] = feat[noise_to_be_chosen1]
#         x[token_nodes1] += self.enc_mask_token1

#         h2 = self.dec(graph, diff_graph, x, edge_weight)
#         loss1 = self.criterion(h1[mask_nodes], h2[mask_nodes])

#         return loss1

#     def get_embed(self, graph, diff_graph, feat, edge_weight):
#         h1 = self.enc(graph, diff_graph, feat, edge_weight)
#         h2 = self.dec(graph, diff_graph, feat, edge_weight)

#         return h1, h2