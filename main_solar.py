import argparse
import torch
import os
from torch import nn, optim
import json
import time
import wandb
import numpy as np
import sys
from solar.simulator import base_classes

from solar.dataset import SolarSystemDataset
from solar.model import EGNN_vel

sys.modules['base_classes'] = base_classes
#from tqdm import tqdm
from tqdm import  tqdm

parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--lr', type=float, default=1e-6, metavar='N',
                    help='learning rate')
parser.add_argument('--nf_edge', type=int, default=64, metavar='N',
                    help='hidden features for edge mlp and messages')
parser.add_argument('--nf_node', type=int, default=64, metavar='N',
                    help='hidden features for node mlp')
parser.add_argument('--nf_coord', type=int, default=64, metavar='N',
                    help='hidden features for coordinate mlp')
parser.add_argument('--nf', type=int, default=0, metavar='N',
                    help='if not 0, overrides nf_edge, nf_node, and nf_coord')
parser.add_argument('--num_vectors', type=int, default=1, metavar='N',
                    help='number of vector channels')
parser.add_argument('--model', type=str, default='egnn_vel', metavar='N',
                    help='available models: gnn, baseline, linear, linear_vel, se3_transformer, egnn_vel, rf_vel, tfn')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--weight_decay', type=float, default=1e-7, metavar='N',
                    help='weight decay')
parser.add_argument('--norm_diff', type=eval, default=True, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='enables checkpoint saving of model')
parser.add_argument('--timestep_pred', type=int, default=1000, metavar='N')
parser.add_argument('--early_stopping', type=int, default=500, metavar='N')
parser.add_argument('--mass_fcn', type=str, default='log', metavar='N')
parser.add_argument('--gradient_clip', type=float, default=0.0, metavar='N')
parser.add_argument('--train_timesteps', type=int, default=14016, metavar='N')
parser.add_argument('--val_timesteps', type=int, default=1752, metavar='N')
parser.add_argument('--test_timesteps', type=int, default=1752, metavar='N')
parser.add_argument('--eval_test', type=eval, default=False, metavar='N')
parser.add_argument('--update_vel', type=eval, default=False, metavar='N',
                    help='update velocity at each step')


time_exp_dic = {'time': 0, 'counter': 0}


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.nf != 0:
    args.nf_edge = args.nf
    args.nf_node = args.nf
    args.nf_coord = args.nf
torch.manual_seed(args.seed)

time_exp_dic = {'time': 0, 'counter': 0}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weighted_mse_loss(pred, true, original):
    mse = torch.sum((pred - true)**2, -1)
    weight = 1/torch.sum((original - true)**2, -1)
    return torch.mean(mse*weight)
    

def process_data(data, edges):
        data = [d.view(-1, d.size(2)) for d in data]
        data = [x.type(torch.FloatTensor) for x in data]
        data = [d.to(device) for d in data]
        loc, vel, mass, loc_end = data
        edges = [edges[0].to(device), edges[1].to(device)]

        vel_feat = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
        if args.mass_fcn == 'sqrt':
            mass_feat = (torch.sqrt(mass)  - torch.mean(torch.sqrt(mass))).detach()
        elif args.mass_fcn == 'log':
            mass_feat = (torch.log(mass) - torch.mean(torch.log(mass))).detach()
        else:
            mass_feat = torch.log(mass).detach()
        nodes = torch.cat([vel_feat, mass_feat], dim=1).to(device)
        rows, cols = edges
        edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1).to(device)
        return nodes, loc.detach(), edges, vel, edge_attr, loc_end

class BaselineModel(nn.Module):
    def __init__(self, vel_mult=10):
        super(BaselineModel, self).__init__()
        self.vel_mult = vel_mult

    def forward(self, nodes, loc, edges, vel, edge_attr):
        return loc + self.vel_mult*vel

def main():
    wandb.init(config=args,
        project="channels_egnn_solar")


    dataset_train = SolarSystemDataset(partition='train', train_timesteps=args.train_timesteps, val_timesteps=args.val_timesteps,
                                       test_timesteps=args.test_timesteps,  timestep_pred=args.timestep_pred)
    dataset_val = SolarSystemDataset(partition='val', train_timesteps=args.train_timesteps, val_timesteps=args.val_timesteps,
                                       test_timesteps=args.test_timesteps,  timestep_pred=args.timestep_pred)
    dataset_test = SolarSystemDataset(partition='test', train_timesteps=args.train_timesteps, val_timesteps=args.val_timesteps,
                                       test_timesteps=args.test_timesteps,  timestep_pred=args.timestep_pred)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = EGNN_vel(in_node_nf=2, in_edge_nf=1, hidden_edge_nf=64,
    #                hidden_node_nf=64,   hidden_coord_nf=64, 
    #                device=device, act_fn=nn.SiLU(),
    #                n_layers=4, coords_weight=1.0,
    #                recurrent=False, norm_diff=False,
    #                tanh=False, num_vectors=1)

    model = EGNN_vel(in_node_nf=2, in_edge_nf=1, hidden_edge_nf=args.nf_edge, 
                        hidden_node_nf=args.nf_node, hidden_coord_nf=args.nf_coord, device=device, n_layers=args.n_layers,
                        recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh, num_vectors=args.num_vectors, update_vel=args.update_vel)
    model = model.to(device)
    print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    wandb.run.summary['model_params'] = num_params
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    results = {'epochs': [], 'losses': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0

    baseline_model = BaselineModel(vel_mult=0.0135*args.timestep_pred).to(device)
    baseline_train_loss = train(baseline_model, optimizer, 0, loader_train, backprop=False)
    baseline_val_loss = train(baseline_model, optimizer, 0, loader_val, backprop=False)
    wandb.run.summary['train/baseline_loss'] = baseline_train_loss
    wandb.run.summary['val/baseline_loss'] = baseline_val_loss

    for epoch in range(0, args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train)
        wandb.log({'epoch': epoch, 'train/loss': train_loss})
        if epoch % args.test_interval == 0:
            val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
            if args.eval_test:
                test_loss = train(model, optimizer, epoch, loader_test, backprop=False)
            else:
                test_loss = 1e8
            results['epochs'].append(epoch)
            #results['losses'].append(test_loss)
            wandb.log({'epoch': epoch, 'val/loss': val_loss, 'val/best_loss':best_val_loss,
                       'test/loss': test_loss,'test/best_loss':best_test_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.eval_test:
                    best_test_loss = test_loss
                best_epoch = epoch
                if args.checkpoint:
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pt'))
            elif epoch - best_epoch > args.early_stopping:
                break
            wandb.run.summary['best_val_loss'] = best_val_loss
            wandb.run.summary['best_test_loss'] = best_test_loss
            wandb.run.summary['best_epoch'] = best_epoch
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" % (best_val_loss, best_test_loss, best_epoch))



def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = loader.dataset.get_edges(batch_size, n_nodes)
        nodes, loc, edges, vel, edge_attr, loc_end = process_data(data, edges)

        optimizer.zero_grad()
        loc_pred = model(nodes, loc, edges, vel, edge_attr)
        
        loss = weighted_mse_loss(loc_pred, loc_end, loc)
        if backprop:
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size
        if batch_idx % args.log_interval == 0 and (args.model == "se3_transformer" or args.model == "tfn"):
            print('===> {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(loader.dataset.partition,
                epoch, batch_idx * batch_size, len(loader.dataset),
                100. * batch_idx / len(loader),
                loss.item()))

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']


if __name__ == "__main__":
     main()