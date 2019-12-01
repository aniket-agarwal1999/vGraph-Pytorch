import torch
import torch_geometric as pyg
from torch_geometric.data import Dataset
import torch_geometric.datasets as datasets
import pandas as pd

def get_cora():
    dataset = datasets.Planetoid(root='./dataset/Cora', name='Cora')
    return dataset.data

def get_citeseer():
    dataset = datasets.Planetoid(root='./dataset/Citeseer', name='CiteSeer')
    return dataset.data

def get_facebook(code):
    '''
    code: which graph to use from the available facebook social circles subgraphs
    options of code : [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
    '''
    assert code in [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
    edge_file = './dataset/Facebook/'+str(code)+'.edges'
    label_file = './dataset/Facebook/'+str(code)+'.circles'   ### Since the circles file basically contains the ground truth

    for 