from qm9 import dataset
from qm9.models import EGNN
import torch
from torch import nn, optim
import argparse
from qm9 import utils as qm9_utils
import utils
import json
import wandb
import numpy as np

parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='qm9/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=128, metavar='N',
                    help='learning rate')
parser.add_argument('--attention', type=int, default=1, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=7, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--property', type=str, default='homo', metavar='N',
                    help='label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                    help='maximum power to take into one-hot features')
parser.add_argument('--dataset_paper', type=str, default="cormorant", metavar='N',
                    help='cormorant, lie_conv')
parser.add_argument('--node_attr', type=int, default=0, metavar='N',
                    help='node_attr or not')
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                    help='weight decay')
parser.add_argument('--num_vectors', type=int, default=1, metavar='N',
                    help='Number of vectors for coordinate update')
parser.add_argument('--update_coords', type=int, default=0, metavar='N',
                    help='Update coordinates or not')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
print(args)

utils.makedir(args.outf)
utils.makedir(args.outf + "/" + args.exp_name)

dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, args.num_workers)
wandb.init(config=args,
    project="channels_egnn_qm9")

# compute mean and mean absolute deviation
meann, mad = qm9_utils.compute_mean_mad(dataloaders, args.property)
wandb.run.summary['property_mean'] = meann
wandb.run.summary['property_mad'] = mad

model = EGNN(in_node_nf=15, in_edge_nf=0, hidden_edge_nf=args.nf, hidden_node_nf=args.nf, hidden_coord_nf=args.nf, device=device, n_layers=args.n_layers, coords_weight=1.0,
             attention=args.attention, node_attr=args.node_attr,
             num_vectors=args.num_vectors, update_coords=args.update_coords)

print(model)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
num_params = sum([np.prod(p.size()) for p in model_parameters])
wandb.run.summary['model_params'] = num_params
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_l1 = nn.L1Loss()


def train(epoch, loader, partition='train'):
    lr_scheduler.step()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()

        else:
            model.eval()

        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)
        nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

        nodes = nodes.view(batch_size * n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        label = data[args.property].to(device, dtype)

        pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)

        if partition == 'train':
            loss = loss_l1(pred, (label - meann) / mad)
            loss.backward()
            optimizer.step()
        else:
            loss = loss_l1(mad * pred + meann, label)

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())


        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % args.log_interval == 0:
            avg_loss = sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, avg_loss))
            wandb.log({"epoch": epoch, "%s/batch_loss" % partition : avg_loss, "%s/batch" % partition : i})
    return res['loss'] / res['counter']


if __name__ == "__main__":


    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
    run_name = str(wandb.run.name)
    checkpoint_path = args.outf + f"/checkpoint_{run_name}.pt"
    for epoch in range(0, args.epochs):
        train_loss = train(epoch, dataloaders['train'], partition='train')
        wandb.log({"epoch": epoch, "train/loss": train_loss})
        if epoch % args.test_interval == 0:
            val_loss = train(epoch, dataloaders['valid'], partition='valid')
            test_loss = train(epoch, dataloaders['test'], partition='test')
            res['epochs'].append(epoch)
            res['losess'].append(test_loss)
            wandb.log({"epoch": epoch, "valid/loss": val_loss, "test/loss": test_loss})

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_test'] = test_loss
                res['best_epoch'] = epoch
                print("New best, saving")
                torch.save({
                    'epoch': epoch,
                    'net_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'test_loss': test_loss,
                    }, checkpoint_path)
            wandb.run.summary['best_val'] = res['best_val']
            wandb.run.summary['best_test'] = res['best_test']
            wandb.run.summary['best_epoch'] = res['best_epoch']
            print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
            print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))


        json_object = json.dumps(res, indent=4)
        with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
            outfile.write(json_object)