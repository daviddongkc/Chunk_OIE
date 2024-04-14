import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import List, Tuple, Dict
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

def convert_to_adj_tensor(adjacency_list, dep_nodes_list, padding_value=0, cuda_device = 0):
    batch_size = len(dep_nodes_list)
    max_sequence_length = max([len(nodes) for nodes in dep_nodes_list])
    if cuda_device >= 0:
        node_tensor = pad_sequence([torch.tensor(x, device=cuda_device) for x in dep_nodes_list],
                                   batch_first=True, padding_value=padding_value)
        adj_tensor = torch.full((batch_size, max_sequence_length, max_sequence_length), padding_value,
                                 dtype=torch.float32, device=cuda_device)
    else:
        node_tensor = pad_sequence([torch.tensor(x) for x in dep_nodes_list],
                                   batch_first=True, padding_value=padding_value)
        adj_tensor = torch.full((batch_size, max_sequence_length, max_sequence_length), padding_value,
                                 dtype=torch.float32)

    for batch_idx, adjacency_matrix in enumerate(adjacency_list):
        for i, j in adjacency_matrix:
            adj_tensor[batch_idx, i, j] = 1


    return adj_tensor, node_tensor

class DepGCN(nn.Module):
    """
    Label-aware Dependency Convolutional Neural Network Layer
    """

    def __init__(self, dep_num, dep_dim, in_features, out_features):
        super(DepGCN, self).__init__()
        self.dep_dim = dep_dim
        self.in_features = in_features
        self.out_features = out_features

        self.dep_embedding = nn.Embedding(dep_num, dep_dim, padding_idx=0)

        self.dep_attn = nn.Linear(dep_dim + in_features, out_features)
        self.dep_fc = nn.Linear(dep_dim, out_features)
        self.relu = nn.ReLU()

    def forward(self, text, dep_mat, dep_labels):
        dep_label_embed = self.dep_embedding(dep_labels)

        batch_size, seq_len, feat_dim = text.shape
        # print(seq_len)

        val_dep = dep_label_embed.unsqueeze(dim=2) # (batch_size, seq_len, 1, dep_dim)
        val_dep = val_dep.repeat(1, 1, seq_len, 1) # (batch_size, seq_len, seq_len, dep_dim)

        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, seq_len, 1) # (batch_size, seq_len, seq_len, token_dim)

        val_sum = torch.cat([val_us, val_dep], dim=-1) # (batch_size, seq_len, seq_len, dep_dim+token_dim)

        r = self.dep_attn(val_sum) # (batch_size, seq_len, seq_len, out_features)

        p = torch.sum(r, dim=-1) # (batch_size, seq_len, seq_len)
        mask = (dep_mat == 0).float() * (-1e30) # (batch_size, seq_len, seq_len)
        p = p + mask
        p = torch.softmax(p, dim=2) # (batch_size, seq_len, seq_len)
        p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim) # (batch_size, seq_len, seq_len, token_dim)

        output = val_us + self.dep_fc(val_dep)
        output = torch.mul(p_us, output) # (batch_size, seq_len, seq_len, out_features)

        output_sum = torch.sum(output, dim=2) # (batch_size, seq_len, out_features)
        output_sum = self.relu(output_sum)

        return output_sum



class ConstGCN(nn.Module):
    """
    Label-aware Constituency Convolutional Neural Network Layer
    """

    def __init__(self, const_num, const_dim, in_features, out_features):
        super(ConstGCN, self).__init__()
        self.const_num = const_num
        self.in_features = in_features
        self.out_features = out_features


        self.const_embedding = nn.Embedding(const_num, const_dim, padding_idx=0)

        self.const_attn = nn.Linear(const_dim + in_features, out_features)
        self.const_fc = nn.Linear(const_dim, out_features)
        self.relu = nn.ReLU()

    def forward(self, text, const_mat, const_labels):
        const_label_embed = self.const_embedding(const_labels)
        const_label_embed = torch.mean(const_label_embed, 2)

        batch_size, seq_len, feat_dim = text.shape

        val_dep = const_label_embed.unsqueeze(dim=2)
        val_dep = val_dep.repeat(1, 1, seq_len, 1)

        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, seq_len, 1)

        val_sum = torch.cat([val_us, val_dep], dim=-1)

        r = self.const_attn(val_sum)

        p = torch.sum(r, dim=-1)
        mask = (const_mat == 0).float() * (-1e30)
        p = p + mask
        p = torch.softmax(p, dim=2)
        p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim)

        output = val_us + self.const_fc(val_dep)
        output = torch.mul(p_us, output)

        output_sum = torch.sum(output, dim=2)
        output_sum = self.relu(output_sum)

        return output_sum
