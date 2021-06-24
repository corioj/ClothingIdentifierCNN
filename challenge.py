'''
EECS 445 - Introduction to Machine Learning
Fall 2020 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge !!!!!!!! USE THIS !!!!!!!!
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Challenge(nn.Module):
    def __init__(self, modelchoice='m1'):
        super().__init__()

        # TODO:
        self.model = None
        
        # input is 3 x 64 x 64
        
        # Different image manipulation/processing ideas?
        
        # First idea: go deeper w/ more layers, use ReLU, Dropout, and BatchNorm layers
        # mess with padding and kernel values later on
        if modelchoice == 'm1':
            self.model = nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = (3,3), padding = 1, stride = 1), # 64 x 64
                nn.ReLU(),
                nn.BatchNorm2d(num_features = 8),
                nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (5,5), padding = 2, stride = 2), # 32 x 32
                nn.ReLU(),
                nn.BatchNorm2d(num_features = 16),
                nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (5,5), padding = 2, stride = 2), # 16 x 16
                nn.ReLU(),
                nn.BatchNorm2d(num_features = 32),
                nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (5,5), padding = 2, stride = 2), # 8 x 8
                nn.ReLU(),
                nn.BatchNorm2d(num_features = 64),
                nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (5,5), padding = 2, stride = 2), # 4 x 4
                nn.ReLU(),
                nn.BatchNorm2d(num_features = 128),
                nn.Flatten(start_dim = 1),
                nn.Linear(in_features = 2048, out_features = 1024),
                nn.ReLU(),
                nn.Dropout(inplace = True),
                nn.Linear(in_features = 1024, out_features = 512),
                nn.ReLU(),
                nn.Dropout(inplace = True),
                nn.Linear(in_features = 512, out_features = 128),
                nn.ReLU(),
                nn.Dropout(inplace = True),
                nn.Linear(in_features = 128, out_features = 64),
                nn.ReLU(),
                nn.Dropout(inplace = True),
                nn.Linear(in_features = 64, out_features = 32),
                nn.ReLU(),
                nn.Dropout(inplace = True),
                nn.Linear(in_features = 32, out_features = 5)
            )

        # Second idea: shorter than #1, similar to but longer than Part 3, except using Batchnorm, Pooling, and Sigmoid
        # The validation
        elif modelchoice == 'm2':
            self.model = nn.Sequential(
                nn.AvgPool2d(kernel_size = (2,2), stride = 2), # 32 x 32
                nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (3,3), padding = 1, stride = 1), # 32 x 32
                nn.Sigmoid(),
                nn.BatchNorm2d(num_features = 16),
                nn.ReLU(),
                nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = (5,5), padding = 2, stride = 2), # 16 x 16
                nn.Sigmoid(),
                nn.BatchNorm2d(num_features = 64),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (5,5), padding = 2, stride = 2), # 8 x 8
                nn.Sigmoid(),
                nn.BatchNorm2d(num_features = 32),
                nn.ReLU(),
                nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = (5,5), padding = 2, stride = 2), # 4 x 4
                nn.Sigmoid(),
                nn.BatchNorm2d(num_features = 16),
                nn.ReLU(),
                nn.Flatten(start_dim = 1),
                nn.Linear(in_features = 256, out_features = 128),
                #nn.BatchNorm1d(num_features = 128),
                nn.Dropout(inplace = True),
                nn.Linear(in_features = 128, out_features = 64),
                #nn.BatchNorm1d(num_features = 64),
                nn.Dropout(inplace = True),
                nn.Linear(in_features = 64, out_features = 32),
                #nn.BatchNorm1d(num_features = 32),
                nn.Dropout(inplace = True),
                nn.Linear(in_features = 32, out_features = 5)
            )
        
        # Third idea: go a little longer than part 3 with Sigmoids, ReLUs, batchnorms
        elif modelchoice == 'm3':
            self.model = nn.Sequential(
                nn.AvgPool2d(kernel_size = (2,2), stride = 2), # 32 x 32
                nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (3,3), stride = (1,1), padding = 1), # 32 x 32
                nn.Sigmoid(),
                nn.BatchNorm2d(num_features = 16),
                nn.ReLU(),
                nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (5,5), stride = (2,2), padding = 2), # 16 x 16
                nn.Sigmoid(),
                nn.BatchNorm2d(num_features = 32),
                nn.ReLU(),
                nn.Conv2d(in_channels = 32, out_channels = 48, kernel_size = (5,5), stride = (2,2), padding = 2), # 8 x 8
                nn.Sigmoid(),
                nn.BatchNorm2d(num_features = 48),
                nn.ReLU(),
                nn.Conv2d(in_channels = 48, out_channels = 16, kernel_size = (5,5), stride = (2,2), padding = 2), # 4 x 4
                nn.Sigmoid(),
                nn.BatchNorm2d(num_features = 16),
                nn.ReLU(),
                nn.Dropout(inplace = True),
                nn.Flatten(start_dim = 1),
                nn.Linear(in_features = 256, out_features = 128),
                nn.Dropout(inplace = True),
                nn.Linear(in_features = 128, out_features = 64),
                nn.Dropout(inplace = True),
                nn.Linear(in_features = 64, out_features = 32),
                nn.Dropout(inplace = True),
                nn.Linear(in_features = 32, out_features = 5)
            )
        
        # Fourth idea: like m3 but much more aggressive pooling/normalization
        elif modelchoice == 'm4':
            self.model = nn.Sequential(
                nn.AvgPool2d(kernel_size = (5,5), stride = 2, padding = 2), # 32 x 32
                nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (5,5), stride = (2,2), padding = 2), # 16 x 16
                nn.Sigmoid(),
                nn.BatchNorm2d(num_features = 16),
                nn.ReLU(),
                nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = (5,5), stride = (2,2), padding = 2), # 8 x 8
                nn.Sigmoid(),
                nn.BatchNorm2d(num_features = 64),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (5,5), stride = (2,2), padding = 2), # 4 x 4
                nn.AvgPool2d(kernel_size = (2,2), stride = 1),
                nn.Flatten(start_dim = 1),
                nn.Linear(in_features = 288, out_features = 128),
                nn.ReLU(),
                nn.Linear(in_features = 128, out_features = 64),
                nn.ReLU(),
                nn.Linear(in_features = 64, out_features = 32),
                nn.ReLU(),
                nn.Linear(in_features = 32, out_features = 5)
            )
        #
        else:
            print('bro... typo :(')
        

        self.init_weights()

    def init_weights(self):
        # TODO:
        
        # First idea: normal initialization
        '''
        for param in self.model.modules():
            if isinstance(param, nn.Conv2d):
                c_in = param.weight.size(1)
                nn.init.normal_(param.weight, 0.0, 1 / sqrt(5*5*c_in))
                nn.init.constant_(param.bias, 0.0)
            if isinstance(param, nn.Linear):
                size = param.weight.size(1)
                nn.init.normal_(param.weight, 0.0, 1/sqrt(size))
                nn.init.constant_(param.bias, 0.0)
        
            
        # Second idea: Kaiming intialization
        '''
        for param in self.model.modules():
            if isinstance(param, nn.Conv2d):
                nn.init.kaiming_uniform_(param.weight, nn.init.calculate_gain(nonlinearity = "relu"))
                nn.init.constant_(param.bias, 0.0)
            if isinstance(param, nn.Linear):
                nn.init.kaiming_uniform_(param.weight, nn.init.calculate_gain(nonlinearity = "relu"))
                nn.init.constant_(param.bias, 0.0)
                
        # Third idea: Xavier initialization
        '''
        for param in self.model.modules():
            if isinstance(param, nn.Conv2d):
                nn.init.xavier_uniform_(param.weight, nn.init.calculate_gain(nonlinearity = "sigmoid"))
                nn.init.constant_(param.bias, 0.0)
            if isinstance(param, nn.Linear):
                nn.init.xavier_uniform_(param.weight, nn.init.calculate_gain(nonlinearity = "sigmoid"))
                nn.init.constant_(param.bias, 0.0)
        '''

    def forward(self, x):
        N, C, H, W = x.shape

        # TODO:
        z = self.model(x)

        #

        return z
