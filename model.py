import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils import negative_sampling

class model(nn.Module):
    def __init__(self, size, categories, embedding_dim=128, negative_sample=False):
        r'''
        size: the number of vertices in the case 
        categories: the number of categories that the nodes need to classified in
        embedding_dim: the dimension for the node and community embedding
        negative_sample: whether to use negative_sample or simple softmax for calculating p(c|z)
        '''
        super().__init__()

        ### These embeddings have been decided by the authors themselves
        # For the case of node embeddings
        self.node_embeddings = nn.Embedding(size, embedding_dim)
        # For the case of community embeddings
        self.community_embedding = nn.Linear(embedding_dim, categories)

        # So as to get back the edges
        self.decode = nn.Linear(embedding_dim, size)

        self.negative_sample = negative_sample

    def negative_sampling_line(z, edge_index, negative_samples = 5):
        r'''
        Parameters:
        z: the sampled community using gumbel softmax reparametrization trick
        edge_index: edges in the graph
        negative_samples: number of negative samples to be used for the optimization

        The function has been inspired from this file: https://github.com/DMPierre/LINE/blob/master/utils/line.py
        '''
        negativenodes = -self.nodes_embeddings(negsamples).to(device)


    def forward(self, w, c, edge_index, negative_samples=5):
        r'''
        Parameters:
        w : the starting node in the edges of the graph
        c : the ending node in the edges of the graph
        edge_index: edges in the graph
        '''
        phi_w = self.node_embeddings(w)
        phi_c = self.node_embeddings(c)
        
        community_embed = self.community_embedding(phi_w * phi_c)

        ### To get the assigned communities we use gumbel softmax
        ### From this we will be sampling to get w
        z = F.gumbel_softmax(logits=community_embed, tau=1, hard=True, dim=-1)

        prior = self.community_embedding(phi_w)
        prior = F.softmax(prior, dim=-1)

        if self.negative_sample == False:
            ## z.shape = [size, categories]   community_embedding.shape = [categories, embedding_dim]
            node_dist = torch.mm(z, self.community_embedding.weight)

            ### This is the prediction of c by using the community embedding
            recon_c = self.decode(node_dist)
        else:
            recon_c = self.negative_sampling_line(z, edge_index, negative_samples)

        return prior, recon_c, z


