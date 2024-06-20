"""Models and functions for training and using NeuroMatch"""
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn


from utils.preprocess import predict_pretrain

# --- Neuro Match Embedding Model with SAGE GNN ---
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
        #--- Properties and object attributes ---
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.aggr = aggr
        self.dropout = dropout
        self.n_layers = n_layers
        self.skip = skip
        self.conv_type = conv_type
        self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,self.n_layers))
        #----

        #--- Defining the multiples NN layers for the forward func ---
        self.pre_mp = nn.Sequential(nn.Linear(self.input_dim,self.hidden_dim))

        self.conv_model = pyg_nn.SAGEConv #If the model were to be changed, it would be here

        #Initialize the convolution list len = n_layers
        self.convs = nn.ModuleList()

        for l in range(self.n_layers):
            #Dealing with the input for skipping layers in learning (see Over-squashing)
            hidden_input_dim = self.hidden_dim * (l + 1)
            self.convs.append(self.conv_model(hidden_input_dim, self.hidden_dim, aggr=self.aggr))

        post_input_dim = self.hidden_dim * (self.n_layers +1)

        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim)
        )

        #-----

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
            x = F.dropout(x, p=self.dropout, training=self.training,)
            emb = torch.cat((emb, x), 1)
            all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1) #all_emb(n+1) = all_emb(n) | x

        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)
        return emb

    def loss(self, pred, label):
        """Negative log likelihood Loss Function"""
        return F.nll_loss(pred, label)


# --- Functions to use with the NM model ---

def nm_criterion(pos_target_emb, pos_query_emb,neg_target_emb, neg_query_emb,margin=0.1):
    """neuromatch criterion impl"""
    e_pos = torch.max(torch.zeros_like(pos_target_emb),pos_query_emb-pos_target_emb)**2 #shape = (32,64)
    e_neg = torch.max(torch.zeros_like(neg_target_emb),neg_query_emb-neg_target_emb)**2 #shape = (32,64)

    e_neg_reduced = torch.sum(e_neg,dim=1) #shape = (32,1)
    e_pos_reduced = torch.sum(e_pos,dim=1) #shape = (32,1)

    neg_loss = torch.sum(torch.max(torch.zeros_like(e_neg_reduced),margin - e_neg_reduced)) #shape = (1,1)
    pos_loss = torch.sum(e_pos_reduced) #shape = (1,1)

    return neg_loss+pos_loss

def faulty_criterion(pos_target_emb, pos_query_emb,neg_target_emb, neg_query_emb,margin=0.1):
    """Misimplemented criterion"""
    e_pos = torch.max(torch.zeros_like(pos_target_emb),pos_query_emb-pos_target_emb)**2 #shape = (32,64)
    e_neg = torch.max(torch.zeros_like(neg_target_emb),neg_query_emb-neg_target_emb)**2 #shape = (32,64)

    neg_loss = torch.sum(torch.max(torch.zeros_like(e_neg),margin - e_neg)) #shape = (1,1)
    pos_loss = torch.sum(e_pos) #shape = (1,1)

    return neg_loss+pos_loss

def treshold_predict(emb_target:torch.Tensor, emb_query:torch.Tensor, treshold:float = 0.1)->bool:
    """Predict if the graph b (corresponding to embedding emb_b) is a subgraph of a (emb_a).
    This version use a simple regular treshold to classify results.

    Parameters
    ----------
    emb_a : `torch.Tensor`
        Embedding of the potential target graph
    emb_b : `torch.Tensor`
        Embedding of the potential subgraph
    treshold : `float`
        Treshold on which to decide if a graph is or not a subgraph (based on emb difference)

    Returns
    -------
    `bool`
        Boolean indicating if 
    """

    e = torch.sum(torch.max(torch.zeros_like(emb_target,
            device=emb_target.device), emb_query - emb_target)**2, dim=1)

    return e < treshold

# --- Classification model

class NeuroMatchPred :   #Refers to the clf_model (and clf_...) in the original git
    """Classification class for Neuromatch, according to the neural-subgraph-learning git implementation
    This class use a basic linear model to decide of the treshold during the learning process"""
    lr = 1e-4
    def __init__(self,lr:float=lr,pretrain:bool=True):
        self.lr = lr
        self.model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.NLLLoss()

        if pretrain:
            predict_pretrain(self)

    def predict(self, emb_target:torch.Tensor, emb_query:torch.Tensor) -> torch.BoolTensor :
        """Predict if the query(s) is a subgraph for the associated target(s).

        Parameters
        ----------
        emb_target : torch.Tensor
            Embedded target graphs.
        emb_query : torch.Tensor
            Embedded Graphs to decide if they are (or not) subgraph of their associated target graph.

        Returns
        -------
        torch.BoolTensor
            Returns the boolean tensor with final classification. True if the graph is a subgraph, False o.w.
        """
        e = self._e(emb_target,emb_query)
        pred = self.model(e)
        pred = pred.argmax(dim=-1).to(torch.bool)
        return pred


    def train(self,emb_target:torch.Tensor, emb_query:torch.Tensor,labels:torch.Tensor) -> Union[torch.FloatTensor, float]:
        """Training process (for a single loop) for the prediction model.

        Parameters
        ----------
        emb_target : torch.Tensor
            Embedded target graphs.
        emb_query : torch.Tensor
            Embedded Graphs to decide if they are (or not) subgraph of their associated target graph.
        labels : torch.Tensor
            Ground truth for the training.

        Returns
        -------
        torch.FloatTensor, float
            returns the prediction tensor of shape (batch_size,2) and the loss of the model
        """
        with torch.no_grad():
            e = self._e(emb_target,emb_query)
        self.model.zero_grad()
        pred = self.model(e)
        clf_loss = self.criterion(pred, labels)
        clf_loss.backward()
        self.opt.step()

        return pred, clf_loss.item()

    def _e(self, emb_target, emb_query):
        """returns a vector of e = max(0, z_q - z_t)**2"""
        return torch.sum(torch.max(torch.zeros_like(emb_target, device=emb_target.device), emb_query - emb_target)**2, dim=1).unsqueeze(1)

    @classmethod
    def hinge_loss(cls,emb_target, emb_query):
        """returns a vector of e = max(0, z_q - z_t)**2"""
        return torch.sum(torch.max(torch.zeros_like(emb_target, device=emb_target.device), emb_query - emb_target)**2, dim=1).unsqueeze(1)

    @classmethod
    def treshold_predict(cls, emb_target:torch.Tensor, emb_query:torch.Tensor, treshold:float = 0.1)->bool:
        """Predict if the graph b (corresponding to embedding emb_b) is a subgraph of a (emb_a).
        This version use a simple regular treshold to classify results.

        Parameters
        ----------
        emb_a : `torch.Tensor`
            Embedding of the potential target graph
        emb_b : `torch.Tensor`
            Embedding of the potential subgraph
        treshold : `float`
            Treshold on which to decide if a graph is or not a subgraph (based on emb difference)

        Returns
        -------
        `bool`
            Boolean indicating if 
        """

        e = torch.sum(torch.max(torch.zeros_like(emb_target,
                device=emb_target.device), emb_query - emb_target)**2, dim=1)

        return e < treshold