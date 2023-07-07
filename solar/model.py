from torch import nn
from torch.nn import functional as F 
from models.gcl import GCL, E_GCL, E_GCL_vel, GCL_rf_vel

class EGNN_vel(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_edge_nf, hidden_node_nf, hidden_coord_nf,
                 device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, recurrent=False,
                 norm_diff=False, tanh=False, num_vectors=1, update_vel=False):
        super(EGNN_vel, self).__init__()
        self.hidden_edge_nf = hidden_edge_nf
        self.hidden_node_nf = hidden_node_nf
        self.hidden_coord_nf = hidden_coord_nf
        self.device = device
        self.n_layers = n_layers
        self.update_vel = update_vel
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_node_nf)
        self.add_module("gcl_%d" % 0, E_GCL_vel(self.hidden_node_nf, self.hidden_node_nf, self.hidden_edge_nf, self.hidden_node_nf, self.hidden_coord_nf, edges_in_d=in_edge_nf, act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh, num_vectors_out=num_vectors))
        for i in range(1, n_layers - 1):
            self.add_module("gcl_%d" % i, E_GCL_vel(self.hidden_node_nf, self.hidden_node_nf, self.hidden_edge_nf, self.hidden_node_nf, self.hidden_coord_nf, edges_in_d=in_edge_nf, act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh, num_vectors_in=num_vectors, num_vectors_out=num_vectors))
        self.add_module("gcl_%d" % (n_layers - 1), E_GCL_vel(self.hidden_node_nf, self.hidden_node_nf,self.hidden_edge_nf, self.hidden_node_nf, self.hidden_coord_nf, edges_in_d=in_edge_nf, act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh, num_vectors_in=num_vectors, last_layer=True))
        self.to(self.device)


    def forward(self, h, x, edges, vel, edge_attr, use_traj=False):
        h = self.embedding(h)
        if use_traj:
            x_traj = [x.clone()]
        for i in range(0, self.n_layers):
            h, x, _, vel_new = self._modules["gcl_%d" % i](h, edges, x, vel, edge_attr=edge_attr)
            if use_traj:
                x_traj.append(x.clone())
            if self.update_vel:
                vel = vel_new
        if use_traj:
            return x.squeeze(2), x_traj
        else:
            return x.squeeze(2)
