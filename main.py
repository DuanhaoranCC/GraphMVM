import torch
import dgl
import time
from model import MG
from torch_geometric import seed_everything
import numpy as np
import warnings
from eval import label_classification
from dataset import process_dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def plot_embeddings(embeddings, data):
    Y = data.detach().cpu().numpy()

    # emb_list = []
    # for k in range(Y.shape[0]):
    #     emb_list.append(embeddings[k])
    emb_list = embeddings

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(Y.shape[0]):
        color_idx.setdefault(Y[i], [])
        color_idx[Y[i]].append(i)
    plt.figure()
    # ax = Axes3D(fig)
    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s=10, alpha=0.7)
    # plt.legend([p3, p4], ['label', 'label1'], loc='lower right', scatterpoints=1)
    plt.savefig("./Our_cite.svg", format='svg')
    plt.show()


def evaluate_cluster(embeds, y, n_label):
    Y_pred = KMeans(n_label, random_state=0).fit(embeds).predict(embeds)
    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)
    sil = silhouette_score(embeds, Y_pred)  #
    ch = calinski_harabasz_score(embeds, Y_pred)  #
    db = davies_bouldin_score(embeds, Y_pred)  #
    return nmi, ari, sil, ch, db


warnings.filterwarnings('ignore')


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = torch.optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = torch.optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = torch.optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = torch.optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return torch.optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


dataname = "pubmed"
label_type = 0
graph, diff_graph, feat, label, train_mask, val_mask, test_mask, \
edge_weight = process_dataset(dataname, 0.01)
n_feat = feat.shape[1]
n_classes = np.unique(label).shape[0]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

graph = graph.to(device)
label = label.to(device)
diff_graph = diff_graph.to(device)
feat = feat.to(device)
edge_weight = torch.tensor(edge_weight).float().to(device)

n_node = graph.number_of_nodes()


def TT(space):
    seed_everything(35536)
    model = MG(feat.size(1), 512, space['p1'], space['p2'], space['rate'], space['rate1'], space['alpha'],
               1, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=space['lr'], weight_decay=space['w'])

    for epoch in range(1, int(space['epoch']) + 1):
        model.train()
        loss = model(graph, diff_graph, feat, edge_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss.item())
        # scheduler.step()

    model.eval()
    z1, z2 = model.get_embed(graph, diff_graph, feat, edge_weight)
    acc = label_classification(z1 + z2, train_mask, val_mask, test_mask,
                               label, label_type, dataname)

    space['acc'] = acc['Acc']['mean']
    print(acc)
    # X = (z1 + z2).detach().cpu().numpy()
    # X = normalize(X, norm='l2')
    # cluster = evaluate_cluster(X, label.cpu().detach().numpy(),
    #                            label.max().cpu().numpy() + 1)
    # plot_embeddings(X, label)
    # print(cluster)
    print(space)


# Cora
# TT({'alpha': 0.6, 'beta': 1.0, 'beta1': 1.0, 'lr': 0.0005, 'n1': 1.0, 'n2': 1.0, 'p1': 0.4, 'p2': 0.4,
#     'rate': 0.1, 'rate1': 0.2, 'acc': 0.852, 'epoch': 500, 'w': 5e-4})
# CiteSeer
# TT({'alpha': 0.5, 'epoch': 700, 'lr': 8e-5, 'p1': 0.8, 'p2': 0.1, 'rate': 0.1, 'rate1': 0.5, 'w': 5e-4, 'acc': 0.733})
# PubMed
TT({'alpha': 0.4, 'epoch': 400, 'lr': 0.0005, 'p1': 0.2, 'p2': 0.6, 'rate': 0.1, 'rate1': 0.5, 'w': 0.0, 'acc': 0.805})

