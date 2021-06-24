"""
EECS 445 - Introduction to Machine Learning
Fall 2020 - Project 2
Fashion Dataset
    Class wrapper for interfacing with the dataset of fashion images
"""
import os
import numpy as np
import pandas as pd
import torch
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import config

def get_train_val_test_loaders(num_classes):
    tr, va, te, _ = get_train_val_test_dataset(num_classes=num_classes)

    batch_size = config('cnn.batch_size')
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)

    return tr_loader, va_loader, te_loader, tr.get_semantic_label

def get_train_val_test_dataset(num_classes=5):
    tr = FashionDataset('train', num_classes)
    va = FashionDataset('val', num_classes)
    te = FashionDataset('test', num_classes)

    # Resize
    tr.X = resize(tr.X)
    va.X = resize(va.X)
    te.X = resize(te.X)
    
    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    tr.X = standardizer.transform(tr.X)
    va.X = standardizer.transform(va.X)
    te.X = standardizer.transform(te.X)
    
    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    tr.X = tr.X.transpose(0,3,1,2)
    va.X = va.X.transpose(0,3,1,2)
    te.X = te.X.transpose(0,3,1,2)
   

    return tr, va, te, standardizer

def get_unlabeled_loader(num_classes):
    unl = get_unlabeled_dataset(num_classes=num_classes)

    batch_size = config('cnn.batch_size')
    unl_loader = DataLoader(unl, batch_size=batch_size, shuffle=False)

    return unl_loader

def get_unlabeled_dataset(num_classes=5):
    tr = FashionDataset('train', num_classes)
    unl = FashionDataset('unlabeled', num_classes)
    unl.X = resize(unl.X)

    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    unl.X = standardizer.transform(unl.X)
    unl.X = unl.X.transpose(0,3,1,2)

    return unl

def resize(X):
    """
    Resizes the data partition X to the size specified in the config file.
    Uses bicubic interpolation for resizing.

    Returns:
        the resized images as a numpy array.
    """
    # TODO: Complete this function
    image_dim = 64
    # create a zeros array of shape (N, input_dim, input_dim, 3)
    images = np.zeros(shape = (X.shape[0], image_dim, image_dim, 3))
    for i in range(X.shape[0]):
        # convert to Pillow Image format
        resized = Image.fromarray(X[i,:,:,:])
        # perform resizing
        resized = resized.resize(size = (image_dim, image_dim), resample = Image.BICUBIC)
        # convert and append to list
        arrResized = np.array(resized).astype('uint8')
        images[i,:,:,:] = arrResized
    
    return images

class ImageStandardizer(object):
    """
    Channel-wise standardization for batch of images to mean 0 and variance 1.
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.

    X has shape (N, image_height, image_width, color_channel)
    """
    def __init__(self):
        super().__init__()
        self.image_mean = None
        self.image_std = None

    def fit(self, X):
        # TODO: Complete this function 
        
        ##### GOTTA BE A VECTORIZED WAY TO DO THIS BRO
        # red channel
        red_mean = np.mean(np.reshape(X[:,:,:,0], -1))
        red_std = np.std(np.reshape(X[:,:,:,0], -1))
        # green channel
        green_mean = np.mean(np.reshape(X[:,:,:,1], -1))
        green_std = np.std(np.reshape(X[:,:,:,1], -1))
        # blue channel
        blue_mean = np.mean(np.reshape(X[:,:,:,2], -1))
        blue_std = np.std(np.reshape(X[:,:,:,2], -1))
        
        # dicts or lists?
        # per channel mean
        self.image_mean = {'red': red_mean, 'green': green_mean, 'blue': blue_mean}
        # per channel std dev
        self.image_std = {'red': red_std, 'green': green_std, 'blue': blue_std}
        #

    def transform(self, X):
        # TODO: Complete this function
        # red 
        for i in range(X.shape[0]):
            X[i,:,:,0] = (X[i,:,:,0] - self.image_mean['red'])/self.image_std['red']
            X[i,:,:,1] = (X[i,:,:,1] - self.image_mean['green'])/self.image_std['green']
            X[i,:,:,2] = (X[i,:,:,2] - self.image_mean['blue'])/self.image_std['blue']
        return X

class FashionDataset(Dataset):

    def __init__(self, partition, num_classes=10):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()

        if partition not in ['train', 'val', 'test', 'unlabeled']:
            raise ValueError('Partition {} does not exist'.format(partition))

        np.random.seed(0)
        self.partition = partition
        self.num_classes = num_classes

        # Load in all the data we need from disk
        self.metadata = pd.read_csv(config('csv_file'))
        self.X, self.y = self._load_data()

        self.semantic_labels = dict(zip(
            self.metadata['numeric_label'],
            self.metadata['semantic_label']
        ))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()

    def _load_data(self):
        """
        Loads a single data partition from file.
        """
        print("loading %s..." % self.partition)

        if self.partition == 'test':
            if self.num_classes == 5:
                df = self.metadata[self.metadata.partition == self.partition]
            else:
                raise ValueError('Unsupported test partition: num_classes must be 5')
        else:
            df = self.metadata[
                (self.metadata.numeric_label < self.num_classes) &
                (self.metadata.partition == self.partition)
            ]

        X, y = [], []
        for i, row in df.iterrows():
            image = imread(os.path.join(config('image_path'), row['filename']))
            X.append(image)
            y.append(row['numeric_label'])

        return np.array(X), np.array(y)

    def get_semantic_label(self, numeric_label):
        """
        Returns the string representation of the numeric class label (e.g.,
        the numberic label 1 maps to the semantic label 'sweaters').
        """
        return self.semantic_labels[numeric_label]

if __name__ == '__main__':
    ## Future note: check scipy imread and imresize
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    np.set_printoptions(precision=3)
    tr, va, te, standardizer = get_train_val_test_dataset()
    print("Train:\t", len(tr.X))
    print("Val:\t", len(va.X))
    print("Test:\t", len(te.X))
    print("Mean:", standardizer.image_mean)
    print("Std: ", standardizer.image_std)
