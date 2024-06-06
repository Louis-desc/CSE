import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class NeuroMatchNetwork(nn.Module) : # Refers to the SkipLastGNN class from the initial project
    """GNN that embbed a node of a graph. The name is to be changed (once I understand better what each part is doing ...)
    """
    input_dim :int = 1
    hidden_dim :int =64
    output_dim :int =64
    aggr :str ="sum"
    n_layers :int =8                            #Size of the studied neighbourhood
    dropout :float =0.0                         #Probably to be deleted
    skip :str ="learnable"                      #Probably to be deleted
    conv_type :str = "SAGE"                     #Probably to be deleted

    def __init__(self, input_dim :int=1, hidden_dim :int=64, output_dim :int=64, aggr :str="sum",
                 n_layers :int=8,dropout :float=0.0,skip :str="learnable",conv_type :str= "SAGE") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.aggr = aggr
        self.dropout = dropout
        self.n_layers = n_layers
        self.skip = skip
        self.conv_type = conv_type
        self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,self.n_layers))
        #----------

        self.pre_mp = nn.Sequential(nn.Linear(self.input_dim,self.hidden_dim))

        #----------
        self.conv_model = pyg_nn.SAGEConv
        #Initialize the convolution list len = n_layers

        self.convs = nn.ModuleList()
        for l in range(self.n_layers):
            hidden_input_dim = self.hidden_dim * (l + 1)
            self.convs.append(self.conv_model(hidden_input_dim, self.hidden_dim, aggr=self.aggr))

        post_input_dim = self.hidden_dim * (self.n_layers +1)
        #-----

        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, self.hidden_dim), 
            nn.Dropout(self.dropout),                      
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim))

    def forward(self,data) :
        """Forward method of Neural Network module"""
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.pre_mp(x)
        all_emb = x.unsqueeze(1)
        emb = x

        for i, conv in enumerate(self.convs) :
            skip_vals = self.learnable_skip[i,:i+1].unsqueeze(0).unsqueeze(-1)
            curr_emb = all_emb * torch.sigmoid(skip_vals)
            curr_emb = curr_emb.view(x.size(0), -1)
            x = conv(curr_emb, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training,)    #What is this dropout use for ?
            emb = torch.cat((emb, x), 1)
            all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1) #all_emb(n+1) = all_emb(n) | x

        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)
        return emb

    def loss(self, pred, label):
        """Negative log likelihood Loss Function"""
        return F.nll_loss(pred, label)


def nm_criterion(pos_target_emb, pos_query_emb,neg_target_emb, neg_query_emb,margin=0.1):
    """neuromatch criterion impl"""
    e_pos = torch.max(torch.zeros_like(pos_target_emb),pos_query_emb-pos_target_emb)**2
    e_neg = torch.max(torch.zeros_like(neg_target_emb),neg_query_emb-neg_target_emb)**2

    neg_loss = torch.sum(torch.max(torch.zeros_like(e_neg),margin - e_neg))
    pos_loss = torch.sum(e_pos)

    return neg_loss+pos_loss