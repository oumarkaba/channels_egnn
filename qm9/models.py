from models.gcl import E_GCL, unsorted_segment_sum, unsorted_segment_mean, unsorted_segment_sum_vec
import torch
from torch import nn



class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_edge_nf, hidden_node_nf, hidden_coord_nf,
                edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(),
                recurrent=True, coords_weight=1.0, attention=False,
                num_vectors_in=1, num_vectors_out=1,
                update_coords=False, last_layer=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_edge_nf, hidden_node_nf, hidden_coord_nf,
                edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim,
                act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight,
                attention=attention, num_vectors_in=num_vectors_in, num_vectors_out=num_vectors_out,
                last_layer=last_layer)
        self.update_coords = update_coords
        if not self.update_coords:
            del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        coord_matrix = self.coord_mlp(edge_feat).view(-1, self.num_vectors_in, self.num_vectors_out)
        if coord_diff.dim() == 2:
            coord_diff = coord_diff.unsqueeze(2)
            coord = coord.unsqueeze(2).repeat(1, 1, self.num_vectors_out)
        # coord_diff = coord_diff / radial.unsqueeze(1)
        trans = torch.einsum('bij,bci->bcj', coord_matrix, coord_diff)
        trans = trans# * edge_mask
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        if self.last_layer:
            coord = coord.mean(dim=2, keepdim=True) + agg * self.coords_weight
        else:
            coord += agg * self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask
        # Modified to include coordinates
        if self.update_coords:
            coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr



class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_edge_nf, hidden_node_nf, hidden_coord_nf,
                device='cpu', act_fn=nn.SiLU(), n_layers=4,
                coords_weight=1.0,attention=False, node_attr=1,
                num_vectors=1, update_coords=False):
        super(EGNN, self).__init__()
        self.hidden_edge_nf = hidden_edge_nf
        self.hidden_node_nf = hidden_node_nf
        self.hidden_coord_nf = hidden_coord_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_node_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0

        self.add_module("gcl_%d" % 0,
                E_GCL_mask(self.hidden_node_nf, self.hidden_node_nf,
                    self.hidden_edge_nf, self.hidden_node_nf, self.hidden_coord_nf,
                    edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr,
                    act_fn=act_fn, recurrent=True,
                    coords_weight=coords_weight, attention=attention,
                    num_vectors_in=1, num_vectors_out=num_vectors, update_coords=update_coords))
        for i in range(1, n_layers - 1):
            self.add_module("gcl_%d" % i,
                E_GCL_mask(self.hidden_node_nf, self.hidden_node_nf,
                    self.hidden_edge_nf, self.hidden_node_nf, self.hidden_coord_nf,
                    edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr,
                    act_fn=act_fn, recurrent=True,
                    coords_weight=coords_weight, attention=attention,
                    num_vectors_in=num_vectors, num_vectors_out=num_vectors, update_coords=update_coords))
        self.add_module("gcl_%d" %  (n_layers - 1),
            E_GCL_mask(self.hidden_node_nf, self.hidden_node_nf,
                self.hidden_edge_nf, self.hidden_node_nf, self.hidden_coord_nf,
                edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr,
                act_fn=act_fn, recurrent=True,
                coords_weight=coords_weight, attention=attention,
                num_vectors_in=num_vectors, num_vectors_out=1,
                update_coords=update_coords, last_layer=True))


        self.node_dec = nn.Sequential(nn.Linear(self.hidden_node_nf, self.hidden_node_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_node_nf, self.hidden_node_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_node_nf, self.hidden_node_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_node_nf, 1))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, x, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, x, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_node_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)



