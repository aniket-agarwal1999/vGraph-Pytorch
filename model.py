import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils import structured_negative_sampling

class Model(nn.Module):
    def __init__(self, size, categories, embedding_dim=128, negative_sample=False):
        '''
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

        self.initialize()


    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def negative_sampling_line(z, edge_index, num_negative_samples = 5):
        '''
        Parameters:
        z: the sampled community using gumbel softmax reparametrization trick
        edge_index: edges in the graph
        negative_samples: number of negative samples to be used for the optimization

        The function has been partially inspired from this file: https://github.com/DMPierre/LINE/blob/master/utils/line.py
        '''

        ## Basically this will output a tuple of length 3 and the third index will contain the nodes from negative edges
        negsamples = structured_negative_sampling(edge_index)
        negsamples = negsamples[2]

        negativenodes = -self.nodes_embeddings(negsamples)

        mulpositivebatch = torch.mul(v_i, v_j)
        positivebatch = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))


    def forward(self, w, c, edge_index, num_negative_samples=5):
        '''
        Parameters:
        w : the starting node in the edges of the graph
        c : the ending node in the edges of the graph
        edge_index: edges in the graph
        '''
        phi_w = self.node_embeddings(w)
        phi_c = self.node_embeddings(c)
        
        q = self.community_embedding(phi_w * phi_c)

        ### To get the assigned communities we use gumbel softmax
        ### From this we will be sampling to get w
        z = F.gumbel_softmax(logits=q, tau=1, hard=True, dim=-1)

        prior = self.community_embedding(phi_w)
        prior = F.softmax(prior, dim=-1)

        if self.negative_sample == False:
            ## z.shape = [size, categories]   community_embedding.shape = [categories, embedding_dim]
            node_dist = torch.mm(z, self.community_embedding.weight)
            ### This is the prediction of c by using the community embedding
            recon_c = self.decode(node_dist)
        else:
            recon_c = self.negative_sampling_line(z, edge_index, num_negative_samples)

        return prior, recon_c, F.softmax(q, dim=-1)


