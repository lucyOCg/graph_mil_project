#gnn_model from https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataset import PatchDataset
import copy
import torch.nn.functional as F
from torch_geometric.nn import (
    Aggregation,
    MaxAggregation,
    MeanAggregation,
    MultiAggregation,
    SAGEConv,
    SoftmaxAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
)


train_dataset = PatchDataset(root="/nobackup/projects/bdlds05/lucyg/graph_project/10x_res18_histo/", filename = 'graph_raw_10x_res18_train.csv')
test_dataset = PatchDataset(root="/nobackup/projects/bdlds05/lucyg/graph_project/10x_res18_histo/", filename="graph_raw_10x_res18_test.csv", test=True)
val_dataset = PatchDataset(root="/nobackup/projects/bdlds05/lucyg/graph_project/10x_res18_histo/", filename="graph_raw_10x_res18_val.csv", val=True)



def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

        


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, aggr='mean', aggr_kwargs=None):
        super().__init__()
        self.conv1 = SAGEConv(
            train_dataset.num_node_features,
            hidden_channels,
            aggr=aggr,
            aggr_kwargs=aggr_kwargs,
        )
        self.conv2 = SAGEConv(
            hidden_channels,
            num_classes=3,
            out_channels = 512,
            aggr=copy.deepcopy(aggr),
            aggr_kwargs=aggr_kwargs,
        )

    def forward(self, x, label, edge_index, batch):
        #print(batch.size())
        #print(x.size())
        #print(len(x[0]))
        x = self.conv1(x, edge_index)
        #print(x.size())
        x = x.tanh()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        #print(x.size())
        #print(len(x[0]))
        return x
        
        
        
        
"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 512, D = 256, dropout = True, n_classes = 1):#changed L=1024 to L=512
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))
            print('No gating!')
        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 512, D = 256, dropout = False, n_classes = 1):#changed L=1024 to L=512
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
            print('With gating!')

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,#changed gate to true CHANGED N_CLASSES TO 3
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(CLAM_SB, self).__init__()
        #self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict = {"small": [512, 512, 256], "big": [512, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        graph_net = GNN(hidden_channels=256, aggr='mean', aggr_kwargs=None)
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, x, label, edge_index, batch, instance_eval=False, return_features=False, attention_only=False):#attention only is false
        graph_net = GNN(hidden_channels=256)
        h = graph_net(x, label, edge_index, batch)
        device = h.device
        #h = torch.squeeze(h) #added for res18 LMC imagenet features
        #print('size', h.shape)
        #h = torch.squeeze(h) #added for res18 LMC imagenet features
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
                
        M = torch.mm(A, h)
        #print('M', M.shape) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        #print('Y_prob', Y_prob.shape)
        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
