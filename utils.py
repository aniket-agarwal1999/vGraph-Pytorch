import torch
import torch.nn as nn
import torch_geometric as pyg
import torch.nn.functional as F
from sklearn import metrics
from sklearn.cluster import KMeans
import community   ## For calculating modularity

def vGraph_loss(c, recon_c, prior, q):
    '''
    c: The original node value
    recon_c: The reconstructed node value
    prior: p(z|w)
    q = q(z|c,w)
    '''

    BCE_loss = F.cross_entropy(recon_c, c) / c.shape[0]   ### Normalization is necessary or the dimension of c is too large and it will be the most weighted
    # KL_div_loss = F.kl_div(torch.log(prior + 1e-20), q, reduction='batchmean')
    KL_div_loss = torch.sum(q*(torch.log(q + 1e-20) - torch.log(prior)), -1).mean()    ## As such main use is of just mean()
    
    loss = BCE_loss + KL_div_loss
    return loss

def load_checkpoint(ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt

def save_checkpoint(state, save_path):
    '''
    Saving checkpoints(state) at the specified save_path location
    '''
    torch.save(state, save_path)

def similarity_measure(edge_index, w, c, gpu_id):
    '''
    Used for calculating the coefficient alpha in the case of community smoothness loss
    Parameters:
    edge_index: edge matrix of the graph
    w: the starting node values of an edge
    c: the ending node values of an edge
    '''

    alpha = torch.zeros(w.shape[0], 1)
    alpha = cuda(alpha, gpu_id)
    for i in range(w.shape[0]):
        l1 = edge_index[1, :][edge_index[0, :] == w[i]].tolist()
        l2 = edge_index[1, :][edge_index[0, :] == c[i]].tolist()
        
        common_neighbors = [value for value in l1 if value in l2]
        common_neighbors = len(common_neighbors)
        all_neighbors = len(l1) + len(l2)
        similarity = (float)(common_neighbors/all_neighbors)
        alpha[i, 0] = similarity
    
    return alpha

def cuda(xs, gpu_id):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda(int(gpu_id[0]))
        else:
            return [x.cuda(int(gpu_id[0])) for x in xs]
    return xs


def calculate_nonoverlap_losses(model, dataset, edge_index)
    '''
    For calculating losses pertaining to the non-overlapping dataset, namely, Macro F1, Micro F1, Modularity, NMI
    '''
    model.eval()
    labels = dataset.y
    w = edge_index[0, :]
    c = edge_index[1, :]
    _, _, q = model(w, c, edge_index)

    new_labels = torch.zeros(w.shape[0], 1)
    for i in range(w.shape[0]):
        new_labels[i] = labels[w[i]]
    
    kmeans = KMeans(n_clusters=torch.unique(labels).shape[0], random_state=0).fit(q)

    ###For calculating modularity
    assignment = {i: int(kmeans.labels_[i]) for i in range(q.shape[0])}
    networkx_graph = pyg.utils.to_networkx(dataset)
    modularity = community.modularity(assignment, dataset)

    ###For calculating macro and micro F1 score
    macro_F1 = metrics.f1_score(new_labels.numpy(), kmeans.labels_, average='macro')
    micro_F1 = metrics.f1_score(new_labels.numpy(), kmeans.labels_, average='micro')

    return modularity, macro_F1, micro_F1

def calculate_jaccard():
    '''
    ## This is for the overlapping case
    ''' 



def calculate_f1():
    '''
    ## This is for the overlapping case
    '''