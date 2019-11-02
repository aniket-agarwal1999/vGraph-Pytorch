import torch
import torch.nn as nn
import torch_geometric as pyg
import torch.nn.functional as F

def vGraph_loss(c, recon_c, prior, q):
    '''
    c: The original node value
    recon_c: The reconstructed node value
    prior: p(z|w)
    q = q(z|c,w)
    '''

    BCE_loss = F.cross_entropy(recon_c, c)
    KL_div_loss = F.kl_div(torch.log(prior + 1e-20), q, reduction='batchmean')
    
    loss = BCE_loss + KL_div_loss
    return loss

def load_checkpoint(ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt

def similarity_measure(edge_index, w, c):
    '''
    Used for calculating the coefficient alpha in the case of community smoothness loss
    Parameters:
    edge_index: edge matrix of the graph
    w: the starting node values of an edge
    c: the ending node values of an edge
    '''

    alpha = torch.zeros(w.shape[0], 1)
    for i in range(w.shape[0]):
        l1 = edge_index[1, :][edge_index[0, :] == w[i]].tolist()
        l2 = edge_index[1, :][edge_index[0, :] == c[i]].tolist()
        
        common_neighbors = [value for value in l1 if value in l2]
        common_neighbors = len(common_neighbors)
        all_neighbors = len(l1) + len(l2)
        similarity = (float)(common_neighbors/all_neighbors)
        alpha[i, 0] = similarity
    
    return alpha


    
    