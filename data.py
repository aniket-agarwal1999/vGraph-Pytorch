import torch
import torch_geometric as pyg
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import torch_geometric.datasets as datasets
import pandas as pd
import re

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

    starting_node = []
    ending_node = []

    with open(edge_file, 'r') as f:
        for line in f:
            li = line.strip().split()
            starting_node.append(int(li[0]))
            ending_node.append(int(li[1]))

    edge_index = torch.zeros([2, len(starting_node)])
    edge_index[0, :] = torch.tensor(starting_node)
    edge_index[1, :] = torch.tensor(ending_node)
    edge_index = edge_index.long()

    communities = []
    with open(label_file, 'r') as f:
        for line in f:
            nodes = re.split(' |\t', line.strip())[1:]
            communities.append([x for x in nodes])
    
    return edge_index, communities