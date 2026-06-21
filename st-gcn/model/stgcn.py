# Spatial temporal graph convolutional neural network
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgcn import unit_sgcn
from tgcn import unit_tgcn

default_layer_config = [(64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1),
                        (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)] # (in_channles, out_channels, temporal_stride)

class stgcn(nn.Module):
    def __init__(self, 
                 num_class, 
                 window_size, 
                 num_point, 
                 in_channels=2, 
                 layer_config=default_layer_config, 
                 graph=None, 
                 graph_config=None, 
                 temporal_kernel_size=9, 
                 dropout=0.5, 
                 non_linearity='relu', 
                 learnable_mask=False,
                 learnable_edges=False):
        """Spatial temporal graph convolutional neural network

        Args:
            num_class (int): total number of classes
            window_size (int): length of input sequence
            num_point (int): number of human body keypoints(number of nodes in a spatial graph)
            in_channels (int, optional): input feature vector. Defaults to 2 with the assumption that the input feature is a xy-coordinate.
            layer_config (List, optional): internal layers configuration. Defaults to default_layer_config.
            graph (_type_, optional): input human skeleton. Defaults to None.
            graph_config (_type_, optional): graph configuration. Defaults to None.
            temporal_kernel_size (int, optional): The kernel size for the 2D convolution in spatial dimension. Defaults to 1.
            dropout (float, optional): Dropout probability after a stgcn unit. Defaults to 0.5.
            non_linearity (str, optional): Non-linearity. Defaults to 'relu'.
            learnable_mask (bool, optional): If true, the adjacency matrix is weighted by a learned mask. Defaults to False.

        Raises:
            ValueError: the input graph cannot be None
        """
        super(stgcn, self).__init__()
        
        if graph is None:
            raise ValueError("Graph is missing")
        else:
            self.A = torch.tensor(graph, dtype=torch.float32, requires_grad=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.A = self.A.to(device)
        
        self.in_channels = in_channels
        self.num_class = num_class
        self.window_size = window_size
        self.num_point = num_point
        self.layer_config = layer_config
        self.graph_config = graph_config
        self.temporal_kernel_size = temporal_kernel_size
        self.learnable_mask = learnable_mask
        self.learnable_edges = learnable_edges
        
        # input layer
        self.stgcn_in = nn.Sequential(
            unit_sgcn(self.A, in_channels, layer_config[0][0], non_linearity=non_linearity, learnable_mask=learnable_mask, learnable_edges=learnable_edges),
            unit_tgcn(layer_config[0][0], layer_config[0][0], kernel_size=temporal_kernel_size, non_linearity=non_linearity)
        )
        
        # internal layers
        self.layers = nn.ModuleList([
            stgcn_unit(self.A, in_channels, out_channels, temporal_kernel_size=temporal_kernel_size, 
                       stride=stride, dropout=dropout, non_linearity=non_linearity, learnable_mask=learnable_mask, learnable_edges=learnable_edges)
            for in_channels, out_channels, stride in layer_config
        ])
        
        # output layer
        self.fcn = nn.Conv1d(layer_config[-1][1], num_class, kernel_size=1)
        nn.init.kaiming_normal_(self.fcn.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        # batch_size, in_channels, frame, vertex, num_people
        # N, C, T, V, M = x.shape
        
        N, C, T, V = x.shape
        
        # from (N, C, T, V, M) to (N*M, C, T, V)
        # x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        x = self.stgcn_in(x) 

        for layer in self.layers:
            x = layer(x) # (N, C, T, V)
        
        # V pooling
        # from (N, C, T, V) to (N, C, T)
        x = torch.mean(x, dim=2)
        
        # M pooling
        # from (N*M, C, T) to (N, C, T)
        # x = x.view(N, M, x.shape[1], x.shape[2])
        # x = x.mean(dim=1)

        # T pooling
        # from (N, C, T) to (N, C, 1)
        # this is a global pooling so that the network can handle different T
        x = torch.mean(x, dim=2, keepdim=True)

        # compute logits
        # from (N, C, 1) to (N, num_class)
        x = self.fcn(x)
        x = F.avg_pool1d(x, x.shape[2:])
        x = x.view(N, self.num_class)
        x = F.softmax(x, dim=1)

        return x

class flow_stgcn(nn.Module):
    def __init__(self, 
                 num_class, 
                 window_size, 
                 num_point, 
                 in_channels=2, 
                 in_flow_channels=64,
                 layer_config=default_layer_config, 
                 graph=None, 
                 graph_config=None, 
                 temporal_kernel_size=9, 
                 dropout=0.5, 
                 non_linearity='relu', 
                 learnable_mask=False,
                 flow_norm=False):
        """Spatial temporal graph convolutional neural network

        Args:
            num_class (int): total number of classes
            window_size (int): length of input sequence
            num_point (int): number of human body keypoints(number of nodes in a spatial graph)
            in_channels (int, optional): input feature vector. Defaults to 2 with the assumption that the input feature is a xy-coordinate.
            layer_config (List, optional): internal layers configuration. Defaults to default_layer_config.
            graph (_type_, optional): input human skeleton. Defaults to None.
            graph_config (_type_, optional): graph configuration. Defaults to None.
            temporal_kernel_size (int, optional): The kernel size for the 2D convolution in spatial dimension. Defaults to 1.
            dropout (float, optional): Dropout probability after a stgcn unit. Defaults to 0.5.
            non_linearity (str, optional): Non-linearity. Defaults to 'relu'.
            learnable_mask (bool, optional): If true, the adjacency matrix is weighted by a learned mask. Defaults to False.

        Raises:
            ValueError: the input graph cannot be None
        """
        super(flow_stgcn, self).__init__()
        
        if graph is None:
            raise ValueError("Graph is missing")
        else:
            assert layer_config[0][0] == in_flow_channels
            
            self.A = torch.tensor(graph, dtype=torch.float32, requires_grad=False)
            self.A_flow = torch.zeros((len(self.A) + 1, num_point + 1, num_point + 1), dtype=torch.float32, requires_grad=False)
            self.A_flow[:2, :num_point, :num_point] = self.A
            self.A_flow[2, num_point - 1, :num_point] = 1
            self.A_flow[2, :num_point, num_point - 1] = 1
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.A = self.A.to(device)
            self.A_flow = self.A_flow.to(device)
        
        self.in_channels = in_channels
        self.in_flow_channels = in_flow_channels
        self.num_class = num_class
        self.window_size = window_size
        self.num_point = num_point
        self.layer_config = layer_config
        self.graph_config = graph_config
        self.temporal_kernel_size = temporal_kernel_size
        self.learnable_mask = learnable_mask
        self.flow_norm = flow_norm
        
        if self.flow_norm:
            self.flows_bn = nn.BatchNorm1d(in_flow_channels)
        
        # input layer
        self.stgcn_in = nn.Sequential(
            unit_sgcn(self.A, in_channels, layer_config[0][0], non_linearity=non_linearity, learnable_mask=learnable_mask),
            unit_tgcn(layer_config[0][0], layer_config[0][0], kernel_size=temporal_kernel_size, non_linearity=non_linearity)
        )
        
        # internal layers
        self.layers = nn.ModuleList([
            stgcn_unit(self.A_flow, in_channels, out_channels, temporal_kernel_size=temporal_kernel_size, 
                       stride=stride, dropout=dropout, non_linearity=non_linearity, learnable_mask=learnable_mask)
            for in_channels, out_channels, stride in layer_config
        ])
        
        # output layer
        self.fcn = nn.Conv1d(layer_config[-1][1], num_class, kernel_size=1)
        nn.init.kaiming_normal_(self.fcn.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x, flows):
        # x: batch_size, in_channels, num_frames, num_vertices
        # flows: batch_size, in_flow_channels, num_frames
        
        N, C, T, V = x.shape
        
        x = self.stgcn_in(x)
        if self.flow_norm:
            flows = self.flows_bn(flows)
        x = torch.cat((x, flows.unsqueeze(-1)), dim=-1) # concate optical flow

        for layer in self.layers:
            x = layer(x) # (N, C, T, V + 1)
        
        # V pooling
        # from (N, C, T, V + 1) to (N, C, T)
        x = torch.mean(x, dim=2)
        
        # T pooling
        # from (N, C, T) to (N, C, 1)
        # this is a global pooling so that the network can handle different T
        x = torch.mean(x, dim=2, keepdim=True)

        # compute logits
        # from (N, C, 1) to (N, num_class)
        x = self.fcn(x)
        x = F.avg_pool1d(x, x.shape[2:])
        x = x.view(N, self.num_class)
        x = F.softmax(x, dim=1)

        return x

class stgcn_unit(nn.Module):
    def __init__(self, A, in_channels, out_channels, temporal_kernel_size=9, stride=1, dropout=0.5, non_linearity='relu', learnable_mask=False, learnable_edges=False):
        """a unit layer of spatial temporal neural network. It performs convolution first in spatial and 
        temporal dimension sequentially. The spatial convolution may change the number channels, whereas
        the temporal convolution is assumed to keep the input tensor dimension unchanged. 

        Args:
            A: The adjacency matrices that represents different partioning of the graph nodes. 
                A has shape (n_p, num_node, num_node), n_p is the number of node partioning.
            in_channels (int): The length of the input feature vector of each node
            out_channels (int): The length of the output feature vector of each node
            temporal_kernel_size (int, optional): The kernel size for the 2D convolution in spatial dimension. Defaults to 1.
            stride (int, optional): The stride for the 2D convolution in the temporal dimension. Defaults to 1. 
            dropout (float, optional): Dropout probability after a stgcn unit. Defaults to 0.5.
            non_linearity (str, optional): Non-linearity. Defaults to 'relu'.
            learnable_mask (bool, optional): If true, the adjacency matrix is weighted by a learned mask. Defaults to False.
        """
        super(stgcn_unit, self).__init__()
        
        self.sgcn = unit_sgcn(A, in_channels, out_channels, non_linearity=non_linearity, learnable_mask=learnable_mask, learnable_edges=learnable_edges) # by default use stride = 1
        self.tgcn = unit_tgcn(out_channels, out_channels, kernel_size=temporal_kernel_size, stride=stride, non_linearity=non_linearity)
        
        self.dropout = nn.Dropout(dropout)
        
        # sampling unit to transform input x to match the dimension of 
        # the output of one stgcn layer to achieve skip 
        # TODO: check for ways to do skip connection and whether apply layer/batch norm
        if in_channels != out_channels or stride != 1:
            self.transform = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
            
            # alternatively, this is what the original paper does
            # self.transform = tgcn(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.transform = None
        
    def forward(self, x):
        if self.transform:
            xt = self.transform(x)
        else:
            xt = x
        
        y = self.sgcn(x)
        y = self.tgcn(y)
        y = self.dropout(y)

        y = y + xt
        
        return y