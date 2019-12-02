import torch
import torch_geometric as pyg
import torch.nn as nn
from model import Model
import argparse
from data import *
import os
import utils
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--negative_sample', type=bool, default=False)   ## If we want to use negative sampling or simply use softmax
parser.add_argument('--decay_epoch', type=int, default=100)
parser.add_argument('--lamda', type=float, default=100.0)   ## For the smoothness trick
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--dataset', type=str, choices=['Cora', 'Citeseer'], default='Cora')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard_curves')


if __name__ == '__main__':
    args = parser.parse_args()

    embedding_dim = args.embedding_dim

    if args.dataset == 'Cora':
        dataset = get_cora()
    elif args.dataset == 'Citeseer':
        dataset = get_citeseer()

    ## For defining the model
    size = dataset.edge_index.shape[1]
    categories = torch.unique(dataset.y).shape[0]

    edge_index = dataset.edge_index
    edge_index = utils.cuda(edge_index, args.gpu_id)

    ## For visualization of loss curves
    if not os.path.isdir(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)
    writer_tensorboard = SummaryWriter(args.tensorboard_dir + '/latest_model_'+args.dataset)

    ## Model for embedding and stuff
    model = Model(size=size, categories=categories, embedding_dim=128, negative_sample=False)
    model = utils.cuda(model, args.gpu_id)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    ### For annealing the learning rate
    lambda1 = lambda lr: 0.99*lr
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    try:
        ckpt = utils.load_checkpoint(args.checkpoint_dir + '/latest_model_' + args.dataset)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    except:
        print(' [*] No checkpoint!')
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        
        optimizer.zero_grad()
        model.train()

        w = torch.cat((edge_index[0, :], edge_index[1, :]))
        c = torch.cat((edge_index[1, :], edge_index[0, :]))

        prior, recon_c, q = model(w, c, edge_index)
        
        ### vGraph loss
        vgraph_loss = utils.vGraph_loss(c, recon_c, prior, q)

        ### Now we will enforce community-smoothness regularization
        ### So we need d(p(z|c), p(z|w)), where p(z|w)=prior and p(z|c) can be easily calculated from this
        prior_c = torch.cat((prior[prior.shape[0]//2:, :], prior[0:prior.shape[0]//2, :]))

        d = (prior_c - prior)**2
        alpha = utils.similarity_measure(edge_index, w, c, args.gpu_id)

        regularization_loss = alpha*d
        regularization_loss = regularization_loss.mean()

        total_loss = vgraph_loss + args.lamda*regularization_loss

        total_loss.backward()
        optimizer.step()

        print('Epoch: ', epoch+1, ' done!!')
        print('Total error: ', total_loss)

        if epoch % 100 == 0:
            lr_scheduler.step()
            # modularity, macro_F1, micro_F1 = utils.calculate_nonoverlap_losses(model, dataset, edge_index)
            # f = open(args.dataset + '_results.txt', 'a+')
            # f.write('Epoch :', epoch, ' modularity: ', modularity, ' macro_F1: ', macro_F1, ' micro_F1: ', micro_F1, ' \n')

        
        writer_tensorboard.add_scalars('Total Loss', {'vgraph_loss':vgraph_loss, 'regularization_loss':regularization_loss}, epoch)

        ### Saving the checkpoint
        utils.save_checkpoint({'epoch':epoch+1,
                               'model':model.state_dict(),
                               'optimizer':optimizer.state_dict()},
                               args.checkpoint_dir + '/latest_model_'+args.dataset+'.ckpt')
    
    
    writer_tensorboard.close()

