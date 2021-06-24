"""
EECS 445 - Introduction to Machine Learning
Fall 2020 - Project 2
Train Autoencoder
    Trains an autoencoder to learn a sparse representation of images data
    Periodically outputs training information, and saves model checkpoints
    Usage: python train_autoencoder.py
"""
import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from model.autoencoder import Autoencoder, AutoencoderClassifier
from train_common import *
from utils import config
import utils

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """

    # TODO: complete the training step
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        
        ######### GO OVER THIS, SOMETHTING IS PROBABLY QUITE WRONG HERE SINCE MYY VALIDATION ERROR IS MARGINALLY BETTER THAN RANDOM CLASSIFICATION
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch, stats):
    """
    Evaluates the `model` on the train and validation set.
    """
    # class is detemined by y_true
    # go thru each value of y_true compare it to indices in X, 
    
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in tr_loader:
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        train_loss = np.mean(running_loss)
        train_acc = correct / total
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        correctc0, totalc0 = 0, 0
        correctc1, totalc1 = 0, 0
        correctc2, totalc2 = 0, 0
        correctc3, totalc3 = 0, 0
        correctc4, totalc4 = 0, 0
        running_loss = []
        for X, y in val_loader:
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
            # class accuracies
            # outerwear
            print('outer')
            totalc0 += (y[y == 0]).size(0)
            print(totalc0)
            correctc0 += ((predicted == y) & (y == 0)).sum().item()
            # sweatshirts
            print('sweat')
            totalc1 += (y[y == 1]).size(0)
            print(totalc1)
            correctc1 += ((predicted == y) & (y == 1)).sum().item()
            # tops
            print('tops')
            totalc2 += (y[y == 2]).size(0)
            print(totalc2)
            correctc2 += ((predicted == y) & (y == 2)).sum().item()
            # pants
            print('pants')
            totalc3 += (y[y == 3]).size(0)
            print(totalc3)
            correctc3 += ((predicted == y) & (y == 3)).sum().item()
            # jeans
            print('jeans time')
            totalc4 += (y[y == 4]).size(0)
            print(totalc4)
            correctc4 += ((predicted == y) & (y == 4)).sum().item()
                
        val_loss = np.mean(running_loss)
        val_acc = correct / total
        print('Outerwear accuracy: ' + str(correctc0/totalc0))
        print('Sweatshirts accuracy: ' + str(correctc1/totalc1))
        print('Tops accuracy: ' + str(correctc2/totalc2))
        print('Pants accuracy: ' + str(correctc3/totalc3))
        print('Jeans accuracy: ' + str(correctc4/totalc4))
    stats.append([val_acc, val_loss, train_acc, train_loss])
    utils.log_cnn_training(epoch, stats)
    utils.update_cnn_training_plot(axes, epoch, stats)

def main():
    # data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        num_classes=config('autoencoder.classifier.num_classes'))

    ae_classifier = AutoencoderClassifier(config('autoencoder.ae_repr_dim'),
        config('autoencoder.classifier.num_classes'))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ae_classifier.parameters(),
        lr=config('autoencoder.classifier.learning_rate'))

    # freeze the weights of the encoder, I CHANGED FORM fc1, fc2
    for name, param in ae_classifier.named_parameters():
        if '.fc_1' in name or '.fc_2' in name or 'fc1' in name or 'fc2' in name:
            param.requires_grad = False

    # Attempts to restore the latest checkpoint if exists
    print('Loading autoencoder...')
    ae_classifier, _, _ = restore_checkpoint(ae_classifier,
        config('autoencoder.checkpoint'), force=True, pretrain=True)
    print('Loading autoencoder classifier...')
    ae_classifier, start_epoch, stats = restore_checkpoint(ae_classifier,
        config('autoencoder.classifier.checkpoint'))

    fig, axes = utils.make_cnn_training_plot(name='Autoencoder Classifier')

    # Evaluate the randomly initialized model
    _evaluate_epoch(axes, tr_loader, va_loader, ae_classifier, criterion,
        start_epoch, stats)

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('autoencoder.classifier.num_epochs')):
        # Train model
        _train_epoch(tr_loader, ae_classifier, criterion, optimizer)

        # Evaluate model
        _evaluate_epoch(axes, tr_loader, va_loader, ae_classifier, criterion,
            epoch+1, stats)

        # Save model parameters
        save_checkpoint(ae_classifier, epoch+1,
            config('autoencoder.classifier.checkpoint'), stats)

    print('Finished Training')

    # Keep plot open
    utils.save_cnn_training_plot(fig, name='ae_clf')
    utils.hold_training_plot()

if __name__ == '__main__':
    main()
