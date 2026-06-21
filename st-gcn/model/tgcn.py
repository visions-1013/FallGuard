# unit temporal graph convolution layer
# perform convolution only in temporal dimension

import torch
import torch.nn as nn
import torch.nn.functional as F

class unit_tgcn(nn.Module):
    def __init__(self, in_channles, out_channels, kernel_size=9, stride=1, non_linearity='relu'):
        """Unit temporal convolutional neural network

        Args:
            in_channles (int): The length of the input feature vector of each node
            out_channels (int): The length of the output feature vector of each node
            kernel_size (int, optional): The kernel size for the 2D convolution. Defaults to 9.
            stride (int, optional): The stride for the 2D convolution. Defaults to 1. 
            dropout (int, optional): _description_. Defaults to 0.
        """
        super(unit_tgcn, self).__init__()
        
        # number of input and output channels
        self.in_channels = in_channles
        self.out_channels = out_channels
        
        # assume stride is 1, compute the padding so that temporal dimension
        # is unchanged after convolution
        padding = (kernel_size - 1) // 2
        
        self.conv_net = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0))
        
        self.bn = nn.BatchNorm2d(self.out_channels)
        
        # ideally always using relu because he initialization is applied as default
        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Sigmoid()
        
        nn.init.kaiming_normal_(self.conv_net.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        # perform convolution in the temporal dimension
        x = self.conv_net(x)
        
        # batch normalization
        x = self.bn(x)
        
        # nonlinearity
        x = self.non_linearity(x)
        
        return x