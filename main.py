import torch
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl
import yaml
import argparse
from model import MG
from torch_geometric import seed_everything
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from dataset import load_graph_classification_dataset

warnings.filterwarnings('ignore')


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std


def collate_fn(batch):
    # graphs = [x[0].add_self_loop() for x in batch]
    graphs = [x[0] for x in batch]
    diff = [x[1] for x in batch]
    labels = [x[2] for x in batch]
    batch_g = dgl.batch(graphs)
    batch_diff = dgl.batch(diff)
    labels = torch.cat(labels, dim=0)
    return batch_g, batch_diff, labels


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    configs = configs[args.dataname]

    for k, v in configs.items():
        if "lr" in k or "w" in k:
            v = float(v)
        setattr(args, k, v)
    return args


parser = argparse.ArgumentParser(description="GraphMVM")
parser.add_argument("--dataname", type=str, default="PROTEINS")
parser.add_argument("--cuda", type=int, default=0)
args = parser.parse_args()
args = load_best_configs(args, "config.yaml")
dataname = args.dataname

graphs, (n_feat, num_classes) = load_graph_classification_dataset(dataname)
train_idx = torch.arange(len(graphs))
batch_size = 256
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')


seed_everything(0)
train_sampler = SubsetRandomSampler(train_idx)
train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn,
                               batch_size=batch_size, pin_memory=True)
eval_loader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=batch_size,
                              shuffle=True)
model = MG(n_feat, 256, args.rate, args.rate1, args.alpha, args.n1, args.n2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w)

for epoch in range(1, args.epoch + 1):
    model.train()
    for g, diff, label in train_loader:
        loss = model(g.to(device), diff.to(device))
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
x_list = []
y_list = []
model.eval()
for g, diff, label in eval_loader:
    z1, z2 = model.get_embed(g.to(device), diff.to(device))
    y_list.append(label.numpy())
    x_list.append((z1 + z2).detach().cpu().numpy())
x = np.concatenate(x_list, axis=0)
y = np.concatenate(y_list, axis=0)
test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
print(test_f1)

# PTC-MR
# TT({'alpha': 0.1, 'epoch': 100, 'lr': 0.0001, 'n1': 5, 'n2': 4, 'rate': 0.3, 'rate1': 0.5, 'w': 0.0, 'acc': 0.65697})
# DD
# TT({'alpha': 0.4, 'epoch': 400, 'lr': 0.0008, 'n1': 3, 'n2': 2, 'rate': 0.4, 'rate1': 0.2, 'w': 0.0, 'acc': 0.79881})
# PROTEINS
# TT({'alpha': 0.2, 'epoch': 800, 'lr': 0.0001, 'n1': 3, 'n2': 2, 'rate': 0.4, 'rate1': 0.8, 'w': 5e-5, 'acc': 0.77247})
# IMDB-BINARY
# TT({'alpha': 0.9, 'epoch': 450, 'lr': 0.0001, 'n1': 4, 'n2': 5, 'rate': 0.8, 'rate1': 0.3, 'w': 1e-5, 'acc': 0.764})
# IMDB-MULTI
# TT({'alpha': 0.5, 'epoch': 400, 'lr': 0.0008, 'n1': 2, 'n2': 5, 'rate': 0.7, 'rate1': 0.6, 'w': 0.0005, 'acc': 0.515})
# COLLAB
# TT({'alpha': 0.8, 'epoch': 900, 'lr': 0.0008, 'n1': 4, 'n2': 4, 'rate': 0.7, 'rate1': 0.8, 'w': 0.0, 'acc': 0.812399})
