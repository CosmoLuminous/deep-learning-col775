# imports
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from src.normalization import NoNorm, BatchNorm, InstanceNorm, LayerNorm, GroupNorm, BatchInstanceNorm

def layer_normalization(dim, norm_type):
    if norm_type == "torch_bn" or norm_type == "inbuilt":
        return nn.BatchNorm2d(dim)

    elif norm_type == "bn":
        return BatchNorm(num_features=dim)

    elif norm_type == "nn":
        return NoNorm()

    elif norm_type == "in":
        return InstanceNorm(num_features=dim)

    elif norm_type == "ln":
        return LayerNorm(num_features=dim)
    
    elif norm_type == "gn":
        return GroupNorm(num_features=dim)

    elif norm_type == "bin":
        return BatchInstanceNorm(num_features=dim)

    else:
        pass

class ResidualBlock(nn.Module):
    """
    Class: ResidualBlock of Resnet Architecture
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None, norm_type="torch_bn"):
        super(ResidualBlock, self).__init__()
        """
        Args:
            channels: Int: Number of Input channels to 1st convolutional layer
            kernel_size: integer, Symmetric Conv Window = (kernel_size, kernel_size)
            filters: python list of integers, defining the number of filters in the CONV layers of the main path
            stride: Int
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.downsample = downsample
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1 = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=stride, 
            padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=1, 
            padding=1, bias=False
        )
        self.relu = nn.ReLU()
        self.norm1 = layer_normalization(out_channels, norm_type)
        self.norm2 = layer_normalization(out_channels, norm_type)
        
    def forward(self, x):
        x_residual = x
        
        if self.downsample is not None:
            x_residual = self.downsample(x)
            
        out = self.norm1(self.conv1(x))
        out = self.relu(out)
        out = self.norm2(self.conv2(out))
        
        out += x_residual
        out = self.relu(out)
        
        return out
        
class ResNet(nn.Module):
    """
    class: Resnet Model with 6n+2 layers for r class classification
    """
    def __init__(self, n_channels = [16, 32, 64], n_layers = [2, 2, 2], n_classes = 10, norm_type = "torch_bn"):
        super(ResNet, self).__init__()
        """
        Input:
        channels: List of channels in residual layers
        n_layers: list of number of residual blocks to be added in the network
        n_classes: number of classes
        norm_type: type of normalization to be used in the network
        """
        
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.norm_type = norm_type
        
        self.conv = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer_norm = layer_normalization(n_channels[0], self.norm_type)
        self.relu = nn.ReLU()
        self.in_channels = n_channels[0]        
        self.out_channels = 0
        self.features = None
                 
        layers = dict()
        for c in range(len(n_channels)):
            layer = list()
            self.out_channels = n_channels[c]
            n = n_layers[c]
            
            for l in range(n):
                downsample = None                
                if self.in_channels != self.out_channels:
                    """CHECK KERNEL SIZE HERE"""
                    downsample = nn.Sequential(
                        nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, 
                                  stride=2, padding=1, bias=False), 
                        layer_normalization(self.out_channels, self.norm_type)
                    )
                if c > 0 and l == 0:
                    stride = 2
                else:
                    stride = 1
                layer.append(ResidualBlock(self.in_channels, self.out_channels, stride = stride, downsample = downsample, norm_type = self.norm_type))
                if l == 0:
                    self.in_channels = self.out_channels       
            layers[c+1] = layer
            
        self.layer1 = nn.Sequential(*layers[1])
        self.layer2 = nn.Sequential(*layers[2])
        self.layer3 = nn.Sequential(*layers[3])
        self.avg_pool = nn.AvgPool2d(kernel_size = 8)
        self.fc = nn.Linear(64, n_classes)
        
    def forward(self, x):
#         print("Input Shape:", x.shape)
        # input convolution
        x = self.layer_norm(self.conv(x))
        x = self.relu(x)
#         print("first conv:", x.shape)
        # residual layers
        x = self.layer1(x)
#         print("layer 1 done:", x.shape)
        x = self.layer2(x)
#         print("layer 2 done:", x.shape)
        x = self.layer3(x)
#         print("layer 3 done:", x.shape)
        
        # average pool
        x = self.avg_pool(x)
        
        # flatten and fc out
        self.features = x.view(-1).detach().cpu()
        x = x.view(-1, 64)
        x = self.fc(x)
        
        return x

    def get_features(self):
        return self.features