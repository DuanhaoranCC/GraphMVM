import torch
import argparse
import warnings
from model import MG
from torch_geometric import seed_everything
import numpy as np
import warnings
import optuna
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from eval import label_classification
from dataset import process_dataset


# torch.autograd.set_detect_anomaly(True)
# seed_everything(65536)
# warnings.filterwarnings('ignore')

dataname = "pubmed"
label_type = 1
graph, diff_graph, feat, label, train_mask, val_mask, test_mask, \
edge_weight = process_dataset(dataname, 0.01)
n_feat = feat.shape[1]
n_classes = np.unique(label).shape[0]
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

graph = graph.to(device)
label = label.to(device)
diff_graph = diff_graph.to(device)
feat = feat.to(device)
edge_weight = torch.tensor(edge_weight).float().to(device)

n_node = graph.number_of_nodes()

def train(trial):
    p2 = trial.suggest_float(name="p2", low=0.0, high=0.9, step=0.1)
    p1 = trial.suggest_float(name="p1", low=0.0, high=0.9, step=0.1)
    b = trial.suggest_float(name="b", low=0.1, high=0.9, step=0.1)
    r1 = trial.suggest_float(name="r1", low=0.1, high=0.9, step=0.1)
    r2 = trial.suggest_float(name="r2", low=0.1, high=0.9, step=0.1)
    lr = trial.suggest_float(name="lr", low=0.00001, high=0.01, step=0.00001)
    epoch = trial.suggest_int(name="epoch", low=100, high=2000, step=100)
    # dim = trial.suggest_int(name="dim", low=128, high=512, step=128)
    # lr = np.around(lr, 3)
    # compensate_rate =trial.suggest_float(name="compensate_rate", low=0.90, high=0.99, step=0.01)
    p1 = np.around(p1, 2)
    p2 = np.around(p2, 2)
    r1 = np.around(r1, 2)
    r2 = np.around(r2, 2)

    seed_everything(35536)
    model = MG(feat.size(1), 256, p1, p2, 1,
               1, r1, r2, b,
               1, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # total_params = sum(p.numel() for p in model.parameters())
    # print("Number of parameter: %.2fM" % (total_params/1e6))
    # scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / 100)) * 0.5
    # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    # model.eval()
    # z1, z2 = model.get_embed(graph, diff_graph, feat, edge_weight)
    # acc = label_classification(z1 + z2, train_mask, val_mask, test_mask,
    #                            label, label_type)['Acc']['mean']
    # print(acc)
    for epoch in range(1, epoch + 1):
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
                               label, label_type, dataname)['Acc']['mean']

    return acc


study = optuna.create_study(direction="maximize")
study.optimize(train, n_trials=100)
print(study.best_params)
