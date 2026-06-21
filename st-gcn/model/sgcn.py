# unit spatial grach convolution layer
# perform convolution only in spatial dimension

import torch
import torch.nn as nn
import torch.nn.functional as F

class unit_sgcn(nn.Module):
    def __init__(self, A, in_channels, out_channels, kernel_size=1, stride=1, non_linearity='relu', learnable_mask=False, learnable_edges=False):
        """Unit spatial graph convolutional neural network

        Args:
            A: The adjacency matrices that represents different partioning of the graph nodes. 
                A has shape (n_p, num_node, num_node), n_p is the number of node partioning.
            in_channels (int): The length of the input feature vector of each node
            out_channels (int): The length of the output feature vector of each node
            kernel_size (int, optional): The kernel size for the 2D convolution. Defaults to 1.
            stride (int, optional): The stride for the 2D convolution. Defaults to 1. 
            learnable_mask (bool, optional): If true, the adjacency matrix is weighted by a learned mask. Defaults to False.
        """
        super(unit_sgcn, self).__init__()
        
        # number of node
        self.V = A.shape[-1]
        
        # adjacency matrix
        self.A = A.view(-1, self.V, self.V)
        self.learnable_edges = learnable_edges
        if learnable_edges:
            self.A = nn.Parameter(A)
        
        # number of adjacency matrix (number of partitions)
        self.num_A = self.A.shape[0]
        
        # number of input and output channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.learnable_mask = learnable_mask
        
        # create num_A kernels for num_A different 
        self.conv_nets = nn.ModuleList([
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1)) for i in range(self.num_A)
        ])
        
        if self.learnable_mask:
            self.mask = nn.Parameter(torch.ones(self.A.shape))
            self.mask_bias = nn.Parameter(torch.zeros(self.A.shape))
            
        self.bn = nn.BatchNorm2d(self.out_channels)
        
        # ideally always using relu because he initialization is applied as default
        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Sigmoid()
        
        # he initialization for relu
        for conv_net in self.conv_nets:
            nn.init.kaiming_normal_(conv_net.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        N, C, T, V = x.shape
        A = self.A
        
        # reweight edges in the graph if learnable mask is enabled
        if self.learnable_mask:
            A = A * self.mask + self.mask_bias
        
        for i, Ai in enumerate(A):
            # aggregate info from neighbors
            x_aggregated = torch.matmul(x.reshape(-1, V), Ai).view(N, C, T, V)

            if i == 0:
                y = self.conv_nets[i](x_aggregated)
            else:
                y += self.conv_nets[i](x_aggregated)
        
        # batch normalization
        y = self.bn(y)
        
        # nonlinearity
        y = self.non_linearity(y)
        
        return y
        