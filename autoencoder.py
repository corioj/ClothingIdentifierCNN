"""
EECS 445 - Introduction to Machine Learning
Fall 2020 - Project 2
Autoencoder
    Constructs a pytorch model for a neural autoencoder
    Autoencoder usage: from model.autoencoder import Autoencoder
    Autoencoder classifier usage:
        from model.autoencoder import AutoencoderClassifier
    Naive method usage: from model.autoencoder import NaiveRecon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Autoencoder(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.repr_dim = repr_dim

        ## Solution: define each layer
        self.pool = nn.AvgPool2d(kernel_size = (2,2), stride = (2,2)) # THIS MAY NEED PADDING???
        self.fc1 = nn.Linear(in_features = 768, out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = 64)
        self.fc3 = nn.Linear(in_features = 64, out_features = 20736)
        ##

        self.deconv = nn.ConvTranspose2d(repr_dim, 3, 5, stride=2, padding=2)
        self.init_weights()

    def init_weights(self):
        # TODO: initialize the parameters for
        #       [self.fc1, self.fc2, self.fc3, self.deconv]
        for layer in [self.fc1, self.fc2, self.fc3]:
            size = layer.weight.size(1)
            nn.init.normal_(layer.weight, 0.0, 0.1**2/sqrt(size))
            nn.init.constant_(layer.bias, 0.01)
         
        nn.init.normal_(self.deconv.weight, 0.0, 0.1**2)
        #

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encoder(self, x):
        # TODO: encoder
        N, C, H, W = x.shape
        # Layer 1
        encoded = self.pool(x)
        # print(encoded.shape) # [128, 3, 16, 16]
        encoded = encoded.view(N, -1)
        # Layer 2
        encoded = self.fc1(encoded)
        encoded = F.elu(encoded)
        # Layer 3
        encoded = self.fc2(encoded)
        encoded = F.elu(encoded)

        return encoded

    def decoder(self, encoded):
        # TODO: decoder
        # Layer 4
        decoded = self.fc3(encoded)
        decoded = F.elu(encoded)
        # Layer 5 & 6
        decoded = self._grow_and_crop(encoded)
        # Layer 7
        decoded = _normalize(decoded)
        return decoded

    def _grow_and_crop(self, x, input_width=18, crop_size=32, scale=2):
        decoded = x.view(-1, self.repr_dim, input_width, input_width)
        decoded = self.deconv(decoded)

        magnified_length = input_width * scale
        crop_offset = (magnified_length - crop_size) // 2
        L, R = crop_offset, (magnified_length-crop_offset)
        decoded = decoded[:, :, L:R, L:R]
        return decoded

class AutoencoderClassifier(nn.Module):
    # skip connections
    def __init__(self, repr_dim, d_out, n_neurons=32):
        super().__init__()
        self.repr_dim = repr_dim

        # TODO: define each layer
        # You will freeze the weights of the autoencoder, connect the output of the encoder to more fully-connected layers,  
        # and then train these last layers on the classification task.
        
        self.pool = nn.AvgPool2d(kernel_size = (2,2), stride = (2,2))
        self.fc1 = nn.Linear(in_features = 768, out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = 64)
        
        # print(repr_dim) 64
        # print(d_out) 5

        self.fc_1 = nn.Linear(repr_dim, n_neurons) # 64 -> 32
        self.fc_2 = nn.Linear(n_neurons, n_neurons) # 32 -> 32
        self.fc_3 = nn.Linear(n_neurons, n_neurons) # 32 -> 32
        self.fc_last = nn.Linear(n_neurons, d_out) # 32 -> 5
        
        # only updating the classifier weights right ie those that are not fc1 or fc2

    def forward(self, x):
        encoded = self.encoder(x)
        
        z1 = F.elu(self.fc_1(encoded))
        z2 = F.elu(self.fc_2(z1))
        z3 = F.elu(self.fc_3(z2))
        z = F.elu(self.fc_last(z1 + z3))
        return z

    def encoder(self, x):
        # TODO: encoder
        N, C, H, W = x.shape
        #
        # print(x.shape) torch.Size([128, 3, 32, 32]) ==> 
        encoded = self.pool(x)
        # view change for nn.Linear layers
        encoded = encoded.view(N, -1)
        # encoder fc layer 1
        encoded = F.elu(self.fc1(encoded))
        # encoder fc layer 2
        encoded = F.elu(self.fc2(encoded))
        # do I maybe return the 
        return encoded

class NaiveRecon(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        compressed = F.avg_pool2d(x, self.scale, stride=self.scale)
        grow = F.interpolate(compressed, size=(32, 32),
            mode='bilinear', align_corners=False)
        reconstructed = _normalize(grow)
        return compressed, reconstructed

def _normalize(x):
    """
    Per-image channelwise normalization
    """
    mean = x.mean(2, True).mean(3, True).mean(0, True)
    std = torch.sqrt((x - mean).pow(2).mean(2, True).mean(3, True).mean(0, True))
    z = (x - mean) / std
    return z
