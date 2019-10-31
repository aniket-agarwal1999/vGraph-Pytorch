import torch
import torch_geometric as pyg
import torch.nn as nn
from model import model
import argparse
from data import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--negative_sample', type=bool, default=False)   ## If we want to use negative sampling or simply use softmax
parser.add_argument('--decay_epoch', type=int, default=100)
parser.add_argument('--lamda', type=float, default=100.0)   ## For the smoothness trick
parser.add_argument('--training', type=bool, default=True)
parser.add_argument('--testing', type=bool, default=False)
parser.add_argument('--validation', type=bool, default=False)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--dataset', type=str, choices=['Cora', 'Citeseer'], default='Cora')




if __name__ == '__main__':
    args = parser.parse_args()

    embedding_dim = args.embedding_dim
    lamda = args.lamda

    if args.training:


