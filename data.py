import torch
import torch_geometric as pyg
from torch_geometric.data import Dataset
import torch_geometric.datasets as datasets

def get_cora():
    dataset = datasets.Planetoid(root='./dataset/Cora', name='Cora')
    return dataset.data

def get_citeseer():
    dataset = datasets.Planetoid(root='./dataset/Citeseer', name='CiteSeer')
    return dataset.data