import copy
import os
import random
import sys
import time
sys.path.append('/home/xiao/Documents/OCW')
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import chain
import json
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

from torch_geometric.nn import knn_graph,radius_graph

# class DynamicEdgeConv(EdgeConv):
#     def __init__(self, in_channels, out_channels, k=6):
#         super().__init__(in_channels, out_channels)
#         self.k = k
#
#     def forward(self, x, batch=None):
#         edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
#         return super().forward(x, edge_index)


class DynamicGAT(pyg_nn.GATConv):

    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(DynamicGAT, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, batch=None):
        if self.k is not None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        else:
            edge_index = radius_graph(x, self.radius, loop=False)

        return super().forward(x, edge_index)

class DynamicGATv2(pyg_nn.GATv2Conv):
    # need to upgrade the version of pytorch geometric to use this pyg_nn.GATv2Conv

    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(DynamicGAT, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, batch=None):
        if self.k is not None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        else:
            edge_index = radius_graph(x, self.radius, loop=False)

        return super().forward(x, edge_index)

class DynamicGCN(pyg_nn.GCNConv):

    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(DynamicGCN, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, batch=None):
        if self.k is not None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        else:
            edge_index = radius_graph(x, self.radius, loop=False)

        return super().forward(x, edge_index)

class DynamicGraphSAGE(pyg_nn.SAGEConv):

    def __init__(self, in_channels, out_channels, k=None, radius=None):
        # default is mean, by changing the aggr to max, it becomes max pooling
        super(DynamicGraphSAGE, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, batch=None):
        if self.k is not None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        else:
            edge_index = radius_graph(x, self.radius, loop=False)

        return super().forward(x, edge_index)

class GraphSAGEPlusPlusDA(pyg_nn.MessagePassing):
    def __init__(self, in_channels, hidden_channels_list, out_channels, k=None, radius=None):
        super(GraphSAGEPlusPlusDA, self).__init__(aggr='mean')  # or other aggregation if needed
        self.k = k
        self.radius = radius
        self.num_layers = len(hidden_channels_list)
        self.convs_mean = nn.ModuleList()
        self.convs_max = nn.ModuleList()

        for i in range(self.num_layers):
            in_channels_layer = in_channels if i == 0 else 2 * hidden_channels_list[i-1]
            out_channels_layer = hidden_channels_list[i]
            self.convs_mean.append(pyg_nn.SAGEConv(in_channels_layer, out_channels_layer, aggr='mean'))
            self.convs_max.append(pyg_nn.SAGEConv(in_channels_layer, out_channels_layer, aggr='max'))

        self.post_mp = nn.Linear(2 * hidden_channels_list[-1], out_channels)

    def reset_parameters(self):
        for conv in self.convs_mean:
            conv.reset_parameters()
        for conv in self.convs_max:
            conv.reset_parameters()
        self.post_mp.reset_parameters()

    def forward(self, x, batch=None):
        # Dynamically create edge_index using knn_graph or radius_graph
        if self.k is not None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        elif self.radius is not None:
            edge_index = radius_graph(x, self.radius, loop=False)
        else:
            raise ValueError("Either k or radius must be provided.")

        all_layers = []
        for i in range(self.num_layers):
            x_mean = self.convs_mean[i](x, edge_index)
            x_max = self.convs_max[i](x, edge_index)

            # Apply ReLU activation function
            x_mean = F.relu(x_mean)
            x_max = F.relu(x_max)

            # Concatenate the mean and max features
            x = torch.cat([x_mean, x_max], dim=1)
            all_layers.append(x)

        # Apply the post message-passing layer
        x_final = self.post_mp(x)
        return F.log_softmax(x_final, dim=-1)

class StaticGraphSAGE(pyg_nn.SAGEConv):
    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(StaticGraphSAGE, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index == None:
            if self.k is not None:
                edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
            else:
                edge_index = radius_graph(x, self.radius, loop=False)
            return super().forward(x, edge_index), edge_index
        else:
            return super().forward(x, edge_index), edge_index
 

class StaticGCN(pyg_nn.GCNConv):
    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(StaticGCN, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index == None:
            if self.k is not None:
                edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
            else:
                edge_index = radius_graph(x, self.radius, loop=False)
            return super().forward(x, edge_index), edge_index
        else:
            return super().forward(x, edge_index), edge_index

class StaticGAT(pyg_nn.GATConv):
    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(StaticGAT, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index == None:
            if self.k is not None:
                edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
            else:
                edge_index = radius_graph(x, self.radius, loop=False)
            return super().forward(x, edge_index), edge_index
        else:
            return super().forward(x, edge_index), edge_index


# ============================================================================
# GraphSAGE++ Variants for Imputation (Static versions)
# ============================================================================

class StaticGraphSAGEPlusDA(nn.Module):
    """
    GraphSAGE++DA: Dual Aggregation (mean + max)
    Combines mean and max aggregation for richer node representations.
    
    Architecture:
        x_mean = SAGEConv(mean)(x, edge_index)
        x_max = SAGEConv(max)(x, edge_index)
        x_out = Linear(concat(x_mean, x_max))
    """
    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(StaticGraphSAGEPlusDA, self).__init__()
        self.k = k
        self.radius = radius
        
        # Dual aggregation: mean and max
        self.conv_mean = pyg_nn.SAGEConv(in_channels, out_channels, aggr='mean')
        self.conv_max = pyg_nn.SAGEConv(in_channels, out_channels, aggr='max')
        
        # Projection layer: concat(mean, max) -> out_channels
        self.projection = nn.Linear(out_channels * 2, out_channels)
        
    def forward(self, x, edge_index=None, batch=None):
        # Build graph if edge_index not provided
        if edge_index is None:
            if self.k is not None:
                edge_index = knn_graph(x, self.k, batch, loop=False)
            else:
                edge_index = radius_graph(x, self.radius, loop=False)
        
        # Dual aggregation
        x_mean = self.conv_mean(x, edge_index)
        x_max = self.conv_max(x, edge_index)
        
        # Concatenate and project
        x_concat = torch.cat([x_mean, x_max], dim=1)
        x_out = F.relu(self.projection(x_concat))
        
        return x_out, edge_index


class StaticGraphSAGEPlusDAC(nn.Module):
    """
    GraphSAGE++DAC: Dual Aggregation + Concat with input
    Adds skip connection by concatenating with original input.
    
    Architecture:
        x_mean = SAGEConv(mean)(x, edge_index)
        x_max = SAGEConv(max)(x, edge_index)
        x_combined = concat(x_mean, x_max)
        x_skip = concat(x_combined, x)  # Skip connection
        x_out = Linear(x_skip)
    """
    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(StaticGraphSAGEPlusDAC, self).__init__()
        self.k = k
        self.radius = radius
        
        # Intermediate hidden dimension
        hidden_dim = out_channels
        
        # Dual aggregation
        self.conv_mean = pyg_nn.SAGEConv(in_channels, hidden_dim, aggr='mean')
        self.conv_max = pyg_nn.SAGEConv(in_channels, hidden_dim, aggr='max')
        
        # Projection with skip connection: concat(mean, max, input) -> out_channels
        self.projection = nn.Linear(hidden_dim * 2 + in_channels, out_channels)
        
    def forward(self, x, edge_index=None, batch=None):
        # Build graph if edge_index not provided
        if edge_index is None:
            if self.k is not None:
                edge_index = knn_graph(x, self.k, batch, loop=False)
            else:
                edge_index = radius_graph(x, self.radius, loop=False)
        
        # Dual aggregation
        x_mean = self.conv_mean(x, edge_index)
        x_max = self.conv_max(x, edge_index)
        
        # Concatenate with skip connection
        x_concat = torch.cat([x_mean, x_max, x], dim=1)
        x_out = F.relu(self.projection(x_concat))
        
        return x_out, edge_index


class StaticGraphSAGEPlusDAMC(nn.Module):
    """
    GraphSAGE++DAMC: Dual Aggregation + Multi-hop Concat
    Uses 2-hop aggregation with multi-scale feature concatenation.
    
    Architecture:
        # First hop
        x1_mean = SAGEConv(mean)(x, edge_index)
        x1_max = SAGEConv(max)(x, edge_index)
        x1 = concat(x1_mean, x1_max)
        
        # Second hop
        x2_mean = SAGEConv(mean)(x1, edge_index)
        x2_max = SAGEConv(max)(x1, edge_index)
        x2 = concat(x2_mean, x2_max)
        
        # Multi-scale concat
        x_out = Linear(concat(x, x1, x2))
    """
    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(StaticGraphSAGEPlusDAMC, self).__init__()
        self.k = k
        self.radius = radius
        
        # Hidden dimension for intermediate layers
        hidden_dim = out_channels // 2 if out_channels >= 64 else 32
        
        # First hop dual aggregation
        self.conv1_mean = pyg_nn.SAGEConv(in_channels, hidden_dim, aggr='mean')
        self.conv1_max = pyg_nn.SAGEConv(in_channels, hidden_dim, aggr='max')
        
        # Second hop dual aggregation (input is concatenated mean+max from hop 1)
        self.conv2_mean = pyg_nn.SAGEConv(hidden_dim * 2, hidden_dim, aggr='mean')
        self.conv2_max = pyg_nn.SAGEConv(hidden_dim * 2, hidden_dim, aggr='max')
        
        # Multi-scale projection: concat(input, hop1, hop2) -> out_channels
        # Dimensions: in_channels + hidden*2 + hidden*2
        self.projection = nn.Linear(in_channels + hidden_dim * 4, out_channels)
        
    def forward(self, x, edge_index=None, batch=None):
        # Build graph if edge_index not provided
        if edge_index is None:
            if self.k is not None:
                edge_index = knn_graph(x, self.k, batch, loop=False)
            else:
                edge_index = radius_graph(x, self.radius, loop=False)
        
        x_input = x  # Save original input for skip connection
        
        # First hop: dual aggregation
        x1_mean = F.relu(self.conv1_mean(x, edge_index))
        x1_max = F.relu(self.conv1_max(x, edge_index))
        x1 = torch.cat([x1_mean, x1_max], dim=1)
        
        # Second hop: dual aggregation on hop1 output
        x2_mean = F.relu(self.conv2_mean(x1, edge_index))
        x2_max = F.relu(self.conv2_max(x1, edge_index))
        x2 = torch.cat([x2_mean, x2_max], dim=1)
        
        # Multi-scale concatenation: original + hop1 + hop2
        x_multi = torch.cat([x_input, x1, x2], dim=1)
        x_out = F.relu(self.projection(x_multi))
        
        return x_out, edge_index



