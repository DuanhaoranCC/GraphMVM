''' Code adapted from https://github.com/kavehhassani/mvgrl '''
import numpy as np
import torch as th
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv
from torch_geometric.datasets import WikiCS
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset, PPIDataset
import networkx as nx

from sklearn.preprocessing import MinMaxScaler

from dgl.nn import APPNPConv
from dgl.transforms import HeatKernel, PPR, DropEdge, AddEdge, FeatMask


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1

def creat_diff(graph, name):
    nx_g = dgl.to_networkx(graph)

    print('computing ppr')
    diff_adj = compute_ppr(nx_g, 0.2)
    print('computing end')

    if name == 'CiteSeer' or name == 'PubMed' or name == 'WikiS' or name == 'computer':
        print('additional processing')
        # feat = th.tensor(preprocess_features(feat.numpy())).float()
        epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
        # epsilons = [1e-5]
        adj = nx.convert_matrix.to_numpy_array(nx_g)
        avg_degree = np.sum(adj) / adj.shape[0]
        epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff_adj >= e).shape[0] / diff_adj.shape[0])
                                      for e in epsilons])]
        diff_adj[diff_adj < epsilon] = 0
        scaler = MinMaxScaler()
        scaler.fit(diff_adj)
        diff_adj = scaler.transform(diff_adj)

    diff_edges = np.nonzero(diff_adj)
    diff_weight = diff_adj[diff_edges]
    diff_graph = dgl.graph(diff_edges)

    return diff_graph, diff_weight

def process_dataset(name, epsilon=0.01):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'Com':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'Photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'CS':
        dataset = CoauthorCSDataset()
    elif name == 'Phy':
        dataset = CoauthorPhysicsDataset()
    elif name == 'WikiCS':
        dataset = WikiCS(root='./Wiki')
        data = dataset[0]
        graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        label = data.y
        diff_graph, diff_weight = creat_diff(graph, name)
        diff_graph.edata['ee'] = th.tensor(diff_weight).float()
        graph = graph.add_self_loop()
        return graph, diff_graph, data.x, label, train_mask, val_mask, test_mask, diff_graph.edata['ee']

    graph = dataset[0]
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    # train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    # val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    # test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    nx_g = dgl.to_networkx(graph)

    print('computing ppr')
    diff_adj = compute_ppr(nx_g, 0.2)
    print('computing end')

    if name == 'citeseer' or name == 'pubmed':
        print('additional processing')
        feat = th.tensor(preprocess_features(feat.numpy())).float()
        epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
        adj = nx.convert_matrix.to_numpy_array(nx_g)
        avg_degree = np.sum(adj) / adj.shape[0]
        epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff_adj >= e).shape[0] / diff_adj.shape[0])
                                      for e in epsilons])]
        diff_adj[diff_adj < epsilon] = 0
        scaler = MinMaxScaler()
        scaler.fit(diff_adj)
        diff_adj = scaler.transform(diff_adj)


    diff_edges = np.nonzero(diff_adj)
    diff_weight = diff_adj[diff_edges]
    diff_graph = dgl.graph(diff_edges)

    # diff_graph.edata['diff'] = th.tensor(diff_weight).float()
    # dgl.save_graphs("./diff_pubmed.bin", diff_graph)

    # diff_graph = dgl.load_graphs("./diff_pubmed.bin")[0][0]
    # diff_weight = diff_graph.edata['diff']
    # transform = PPR(eps=1e-4)
    # transform = HeatKernel(t=2, eps=1e-5)
    # diff_graph = transform(graph)
    # diff_weight = diff_graph.edata['w']
    # transform = DropEdge(p=0.1)
    # transform = AddEdge(ratio=0.3)
    # transform = FeatMask(p=0.5)
    # diff_graph = transform(graph)

    graph = graph.add_self_loop()

    return graph, diff_graph, feat, label, train_mask, val_mask, test_mask, diff_weight


def process_dataset_appnp(epsilon):
    k = 20
    alpha = 0.2
    dataset = PubmedGraphDataset()
    graph = dataset[0]
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    appnp = APPNPConv(k, alpha)
    id = th.eye(graph.number_of_nodes()).float()
    diff_adj = appnp(graph.add_self_loop(), id).numpy()

    diff_adj[diff_adj < epsilon] = 0
    scaler = MinMaxScaler()
    scaler.fit(diff_adj)
    diff_adj = scaler.transform(diff_adj)
    diff_edges = np.nonzero(diff_adj)
    diff_weight = diff_adj[diff_edges]
    diff_graph = dgl.graph(diff_edges)

    return graph, diff_graph, feat, label, train_idx, val_idx, test_idx, diff_weight

# process_dataset('dblp')
