import torch
from model import MG
from torch_geometric import seed_everything
import numpy as np
import warnings
import yaml
import argparse
from eval import label_classification
from dataset import process_dataset

warnings.filterwarnings('ignore')


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
parser.add_argument("--dataname", type=str, default="Cora")
parser.add_argument("--cuda", type=int, default=0)
args = parser.parse_args()
args = load_best_configs(args, "config.yaml")
dataname = args.dataname

label_type = 0
graph, diff_graph, feat, label, train_mask, val_mask, test_mask, \
edge_weight = process_dataset(dataname, 0.01)
n_feat = feat.shape[1]
n_classes = np.unique(label).shape[0]
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

graph = graph.to(device)
label = label.to(device)
diff_graph = diff_graph.to(device)
feat = feat.to(device)
edge_weight = torch.tensor(edge_weight).float().to(device)

n_node = graph.number_of_nodes()


seed_everything(35536)
model = MG(feat.size(1), args.dim, args.p1, args.p2, args.rate, args.rate1, args.alpha).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w)

for epoch in range(1, args.epoch + 1):
    model.train()
    loss = model(graph, diff_graph, feat, edge_weight)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
z1, z2 = model.get_embed(graph, diff_graph, feat, edge_weight)
acc = label_classification(z1 + z2, train_mask, val_mask, test_mask,
                           label, label_type, dataname)

print(acc['Acc']['mean'])
