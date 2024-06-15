import argparse
from functools import partial

from loader import MoleculeDataset
from dataloader import DataLoaderMasking, DataLoaderMaskingPred  # , DataListLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import MVM, GNNDecoder
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from util import MaskAtom

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter

import timeit


class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def train_mae(args, model, loader, optimizer_model, device, alpha_l=1.0, loss_fn="sce"):
    model.train()

    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0
    # a = []
    epoch_iter = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(epoch_iter):
        data, data1 = batch
        data = data.to(device)
        data1 = data1.to(device)
        loss = model(data, data1)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"train_loss: {loss.item():.4f}")
    return loss_accum / step  # , acc_node_accum/step, acc_edge_accum/step


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.35,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_rate1', type=float, default=0.35,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default='', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument('--input_model_file', type=str, default=None)
    parser.add_argument("--alpha_l", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--decoder", type=str, default="gin")
    parser.add_argument("--use_scheduler", action="store_true", default=False)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" % (args.num_layer, args.mask_rate, args.mask_edge))

    dataset_name = args.dataset
    # set up dataset and transform function.
    # dataset = MoleculeDataset("/home/yhkj/dhr/KDD/chem/dataset/" + args.dataset, dataset=args.dataset,
    #                           transform=MaskAtom(num_atom_type=119, num_edge_type=5, mask_rate=args.mask_rate,
    #                                              mask_edge=args.mask_edge))
    dataset = MoleculeDataset("/home/yhkj/dhr/KDD/chem/dataset/" + dataset_name, dataset=dataset_name)

    # loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    loader = DataLoaderMaskingPred(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   mask_rate=args.mask_rate, mask_rate1=args.mask_rate1, mask_edge=args.mask_edge)

    # set up models, one for pre-training and one for context embeddings
    model = MVM(args.num_layer, args.emb_dim, dp=args.dropout_ratio).to(
        device)

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    output_file_temp = "./checkpoints/" + args.output_model_file + f"_{args.gnn_type}"
    lr_scheduler = CosineDecayScheduler(args.lr, 100, args.epochs)
    mm_scheduler = CosineDecayScheduler(1 - 0.9999, 0, args.epochs)
    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))
        # update learning rate
        lr = lr_scheduler.get(epoch - 1)
        for param_group in optimizer_model.param_groups:
            param_group['lr'] = lr
        model.train()
        loss_accum = 0
        # update momentum
        mm = 1 - mm_scheduler.get(epoch - 1)
        epoch_iter = tqdm(loader, desc="Iteration")
        for step, batch in enumerate(epoch_iter):
            data, data1 = batch
            data = data.to(device)
            data1 = data1.to(device)
            loss = model(data, data1)

            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()
            model.update_target_network(mm)
            loss_accum += float(loss.cpu().item())
            epoch_iter.set_description(f"train_loss: {loss.item():.4f}")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), output_file_temp + f"{args.mask_rate}_{epoch}.pth")

    # output_file = "./checkpoints/" + args.output_model_file + f"_{args.gnn_type}"
    # if resume:
    #     torch.save(model.state_dict(), args.input_model_file.rsplit(".", 1)[0] + f"_resume_{args.epochs}.pth")
    # elif not args.output_model_file == "":
    #     torch.save(model.state_dict(), output_file + ".pth")


if __name__ == "__main__":
    main()
