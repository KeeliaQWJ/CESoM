import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from opt_einsum import contract as einsum

def init_lecun_normal(module):
    nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module

def create_custom_forward(module, **kwargs):
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)
    return custom_forward

def activation_checkpointing(function):
    def wrapper(*args, **kwargs):
        if torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(create_custom_forward(function, **kwargs), *args, use_reentrant=False)
        return function(*args, **kwargs)
    return wrapper

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * 2, out_dim),
        )
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        return self.net(x)
    

class AttentiveGRU1(nn.Module):
    """Update node features with attention and GRU.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node (atom) features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        """
        Parameters
        ----------
        g : DGLGraph
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        edge_feats : float32 tensor of shape (E, M1)
            Previous edge features.
        node_feats : float32 tensor of shape (V, M2)
            Previous node features.

        Returns
        -------
        float32 tensor of shape (V, M2)
            Updated node features.
        """
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        # g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class AttentiveGRU2(nn.Module):
    """Update node features with attention and GRU.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node (atom) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        """
        Parameters
        ----------
        g : DGLGraph
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        node_feats : float32 tensor of shape (V, M2)
            Previous node features.

        Returns
        -------
        float32 tensor of shape (V, M2)
            Updated node features.
        """
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        # g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        g.update_all(fn.u_mul_e('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class GetContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        """Edge feature update."""
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        """Edge feature update."""
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])

class GNNLayer(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()
        
        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )

        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)

    def apply_edges(self, edges):
        """Edge feature update by concatenating the features of the destination
        and source nodes."""
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var() 
        g.ndata['hv'] = node_feats 
        g.apply_edges(self.apply_edges) 
        logits = self.project_edge(g.edata['he']) 

        return self.attentive_gru(g, logits, node_feats) # 基于边的特征更新节点特征

class GlobalPool(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GlobalPool, self).__init__()
        
        self.compute_logits = nn.Sequential(
            nn.Linear(node_feat_size + graph_feat_size, 1),
            nn.LeakyReLU()
        )
        
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, graph_feat_size)
        )

        self.gru = nn.GRUCell(graph_feat_size, graph_feat_size)

    def forward(self, g, node_feats, g_feats, get_node_weight=False):
        """
        g : DGLGraph
            Constructed DGLGraphs.
        node_feats : float32 tensor of shape (V, N1)
            Input node features. V for the number of nodes and N1 for the feature size.
        g_feats : float32 tensor of shape (G, N2)
            Input graph features. G for the number of graphs and N2 for the feature size.
        get_node_weight : bool
            Whether to get the weights of atoms during readout.

        Returns
        -------
        float32 tensor of shape (G, N2)
            Updated graph features.
        float32 tensor of shape (V, 1)
            The weights of nodes in readout.
        """
        with g.local_scope():
            g.ndata['z'] = self.compute_logits(
                torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))
            g.ndata['a'] = dgl.softmax_nodes(g, 'z')
            g.ndata['hv'] = self.project_nodes(node_feats)
            context = F.relu(dgl.sum_nodes(g, 'hv', 'a'))
            
            if get_node_weight:
                return self.gru(context, g_feats), g.ndata['a'] # 如果weight为真，则同时返回更新后的图特征和节点权重
            else:
                return self.gru(context, g_feats)
            
class TriangleAttention(nn.Module):
    def __init__(self, d_pair, n_head=4, d_hidden=32, p_drop=0.1, start_node=True):
        super(TriangleAttention, self).__init__()
        self.norm = nn.LayerNorm(d_pair)
        self.to_q = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_pair, n_head*d_hidden)

        self.to_out = nn.Linear(n_head*d_hidden, d_pair)

        self.scaling = 1 / (d_hidden ** 0.5)
        
        self.h = n_head
        self.dim = d_hidden
        self.start_node = start_node
        
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        init_lecun_normal(self.to_b)

        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    @activation_checkpointing
    def forward(self, pair):
        B, N, N, C = pair.shape  

        pair = self.norm(pair)
        
        q = self.to_q(pair).reshape(B, N, N, self.h, -1)  # (B, N, N, h, d_hidden)
        k = self.to_k(pair).reshape(B, N, N, self.h, -1)
        v = self.to_v(pair).reshape(B, N, N, self.h, -1)
        b = self.to_b(pair).reshape(B, N, N, self.h)
        g = torch.sigmoid(self.to_g(pair)).reshape(B, N, N, self.h, -1)
        
        q = q * self.scaling  # (B, N, N, h, d_hidden)
        
        if self.start_node:
            q = q.permute(0, 3, 1, 2, 4)  
            k = k.permute(0, 3, 1, 2, 4)  
            v = v.permute(0, 3, 1, 2, 4)  
            
            attn = torch.matmul(q, k.transpose(-2, -1))  # (B, h, N, N, N)
            attn = attn / math.sqrt(self.dim)
            b = b.permute(0, 3, 1, 2)  # (B, h, N, N)
            attn = attn + b.unsqueeze(-1)  # (B, h, N, N, N)
            
            attn = F.softmax(attn, dim=-1) 
            out = torch.matmul(attn, v)  # (B, h, N, N, d_hidden)
            
            out = out.permute(0, 2, 3, 1, 4) # (B, N, N, h*d_hidden)
        else:
            pass  

        out = g * out  
        out = out.reshape(B, N, N, -1)
        out = self.to_out(out)
        return out  # (B, N, N, d_pair)
    

class AtomProteinAttentionGNN(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 protein_feat_size,
                 graph_feat_size,
                 num_layers,
                 output_size,
                 dropout):
        super(AtomProteinAttentionGNN, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))
        
        self.gnn_layer_complete = GNNLayer(graph_feat_size, graph_feat_size, dropout)

        self.protein_feat_dim_reduction = FeedForward(in_dim=1280, out_dim=protein_feat_size, dropout=dropout)
        self.protein_fc = nn.Linear(protein_feat_size, graph_feat_size)
        n_head = 4
        d_hidden = graph_feat_size // n_head
        self.attention = TriangleAttention(d_pair=graph_feat_size, n_head=n_head, d_hidden=d_hidden, p_drop=dropout, start_node=True)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, batched_graph, batched_graph_c, atom_feats_batch, protein_feats_batch, return_attn_weights=False):
        node_feats = batched_graph.ndata['h']
        edge_feats = batched_graph.edata['e']

        node_feats = self.init_context(batched_graph, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(batched_graph, node_feats)
        
        batched_graph_c.ndata['h'] = node_feats
        node_feats = self.gnn_layer_complete(batched_graph_c, node_feats)
        node_feats_batch_split = torch.split(node_feats, batched_graph.batch_num_nodes().tolist())
        predictions_list = []

        for i in range(len(node_feats_batch_split)):
            node_feats_sample = node_feats_batch_split[i] 
            protein_feats_sample = protein_feats_batch[i]  
            protein_feats_sample = self.protein_feat_dim_reduction(protein_feats_sample) 
            protein_feats_sample = self.protein_fc(protein_feats_sample)  

            mapping_layer = nn.Linear(protein_feats_sample.shape[0], node_feats_sample.shape[0]).to(protein_feats_sample.device)
            protein_feats_mapped = mapping_layer(protein_feats_sample.transpose(0, 1)).transpose(0, 1) 

            node_feats_expanded = node_feats_sample.unsqueeze(1) 
            protein_feats_expanded = protein_feats_mapped.unsqueeze(0)  
            pair = (node_feats_expanded + protein_feats_expanded).unsqueeze(0)  

            attn_output = self.attention(pair)  
            attn_output = attn_output.sum(dim=2)  

            attn_output = attn_output.squeeze(0) 
            fused_feats = node_feats_sample + attn_output

            predictions = self.predict(fused_feats)  
            predictions_list.append(predictions)

        predictions = torch.cat(predictions_list, dim=0)  
        return predictions
    
class Atom_model_fineturn(nn.Module):
    """`Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    Parameters
    ----------
    node_feat_size : int
        Size for the input node (atom) features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    num_layers : int
        Number of GNN layers.
    num_timesteps : int
        Number of timesteps for updating the molecular representation with GRU.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    output_size : int
        Size of the prediction (target labels).
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 graph_feat_size,
                 num_layers,
                 output_size,
                 dropout):
        super(Atom_model_fineturn, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout)) 
#         self.readouts = nn.ModuleList()
#         for t in range(num_timesteps):
#             self.readouts.append(GlobalPool(graph_feat_size, graph_feat_size, dropout))

        self.gnn_layer_ = GNNLayer(graph_feat_size, graph_feat_size, dropout)
    
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, output_size),
            nn.Sigmoid()
        ) 

    def forward(self, g, node_feats, edge_feats,g_c): 
        """
        Parameters
        ----------
        g : DGLGraph
            Constructed DGLGraphs.
        node_feats : float32 tensor of shape (V, N1)
            Input node features. V for the number of nodes and N1 for the feature size.
        edge_feats : float32 tensor of shape (E, N2)
            Input edge features. E for the number of edges and N2 for the feature size.
        get_node_weight : bool
            Whether to get the weights of atoms during readout.

        Returns
        -------
        float32 tensor of shape (G, N3)
            Prediction for the graphs. G for the number of graphs and N3 for the output size.
        node_weights : list of float32 tensors of shape (V, 1)
            Weights of nodes in all readout operations.
        """
        node_feats = self.init_context(g, node_feats, edge_feats) 
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats) 
        
        g_c.ndata['h'] = node_feats
        node_feats = self.gnn_layer_(g_c,node_feats) 

        return self.predict(node_feats) 
    
