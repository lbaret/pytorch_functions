import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np
from sklearn.metrics import accuracy_score


# TODO : Make it general for all wanted scores
# TODO : Remove mean at the return of validation and test functions

# Training function
def train(model: nn.Module, optimizer: Optimizer, loss: nn.Module, train_loader: DataLoader,
          valid_loader: DataLoader = None, epochs: int = 100, gpu: int = None,
          score: list = None, scheduler=None, make_sigmoid=False, make_softmax=False) -> tuple:
    """
    :param model: torch ML model
    :param optimizer: torch optimizer algorithm
    :param loss: loss function
    :param train_loader: training set
    :param valid_loader: validation set
    :param epochs: number of epochs
    :param gpu: gpu number
    :param scheduler: Learning Rate scheduler
    :return: train accuracy, train loss, validation accuracy, validation loss
    """
    # GPU
    if gpu is not None:
        model = model.cuda(gpu)

    epochs_train_loss = []
    epochs_valid_loss = []
    for ep in range(epochs):
        model.training = True

        all_losses = []
        all_predictions = []
        all_targets = []
        for i, (inputs, targets) in enumerate(train_loader):
            # GPU
            if gpu is not None:
                inputs = inputs.cuda(gpu)
                targets = targets.float().cuda(gpu)

            predictions = model(inputs).squeeze()
            err = loss(predictions, targets)

            # Machine is learning
            err.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Clean GPU
            if gpu is not None:
                err = err.detach().cpu()
                inputs = inputs.cpu()
                targets = targets.cpu()
                predictions = predictions.cpu()
                torch.cuda.empty_cache()

            all_losses.append(err)

            if make_sigmoid:
                labels = (F.sigmoid(predictions) >= 0.5) * 1
            elif make_softmax:
                labels = (F.softmax(predictions) >= 0.5) * 1
            else:
                labels = predictions

            all_predictions.append(labels)
            all_targets.append(targets)

            print(
                f'\rBatch : {i + 1} / {len(train_loader)} - Loss : {err:.2e}',
                end='')

        all_predictions = torch.vstack(all_predictions)
        all_targets = torch.vstack(all_targets)

        train_loss = np.vstack(all_losses).mean()

        # Historic
        epochs_train_loss.append(train_loss)

        if scheduler is not None:
            scheduler.step()

        # Validation step
        if valid_loader is not None:
            valid_loss = valid(model, loss, valid_loader, gpu)
            # Historic
            epochs_valid_loss.append(valid_loss)
            print(
                f'\rEpoch : {ep + 1} - Train Loss : {train_loss:.2e} - '
                f'- Valid Loss : {valid_loss:.2e}')
        else:
            # Display epoch information
            print(f'\rEpoch : {ep + 1} - Train Loss : {train_loss:.2e}')

    if valid_loader is not None:
        return epochs_train_loss, epochs_valid_loss

    return epochs_train_loss


# Validation function
def valid(model: nn.Module, loss: nn.Module, valid_loader: DataLoader, gpu) -> tuple:
    """
    :param model: torch ML model
    :param loss: loss function
    :param valid_loader: validation set
    :param gpu: gpu number
    :return: loss, accuracy
    """
    model.training = False
    with torch.no_grad():
        all_losses = []
        all_predictions = []
        all_targets = []
        for i, (inputs, targets) in enumerate(valid_loader):
            if gpu is not None:
                inputs = inputs.cuda(gpu)
                targets = targets.float().cuda(gpu)

            predictions = model(inputs).squeeze()
            err = loss(predictions, targets)

            all_losses.append(err.detach().cpu())

            # Clean GPU
            if gpu is not None:
                err = err.cpu()
                inputs = inputs.cpu()
                targets = targets.cpu()
                predictions = predictions.cpu()
                torch.cuda.empty_cache()

            all_predictions.append((F.sigmoid(predictions) >= 0.5) * 1)
            all_targets.append(targets)

            print(f'\rValid batch : {i + 1} / {len(valid_loader)}', end='')

        all_losses = torch.hstack(all_losses)
        all_predictions = torch.hstack(all_predictions)
        all_targets = torch.hstack(all_targets)

        return all_losses.mean()


# Test
def test(model: nn.Module, loss: nn.Module, test_loader: DataLoader, gpu: int = None) -> tuple:
    """
    :param model: torch ML model
    :param loss: loss function
    :param test_loader: test set (DataLoader)
    :param gpu: gpu number
    :return: loss, accuracy
    """
    model.training = False
    with torch.no_grad():
        all_losses = []
        all_predictions = []
        all_targets = []
        for i, (inputs, targets) in enumerate(test_loader):
            if gpu is not None:
                inputs = inputs.cuda(gpu)
                targets = targets.float().cuda(gpu)

            predictions = model(inputs).squeeze()
            err = loss(predictions, targets)

            all_losses.append(err.detach().cpu())

            # Clean GPU
            if gpu is not None:
                err = err.cpu()
                inputs = inputs.cpu()
                targets = targets.cpu()
                predictions = predictions.cpu()
                torch.cuda.empty_cache()

            all_predictions.append((F.sigmoid(predictions) >= 0.5) * 1)
            all_targets.append(targets)

            print(f'\rTest batch : {i + 1} / {len(test_loader)}', end='')

        all_losses = torch.vstack(all_losses)
        all_predictions = torch.hstack(all_predictions)
        all_targets = torch.hstack(all_targets)

        return all_losses.mean()


# Make predictions
def predict(model: nn.Module, tensor_data: torch.Tensor, gpu: int = None) -> torch.Tensor:
    """
    :param model: torch ML model
    :param tensor_data: tensor of examples
    :param gpu: gpu number (default = None)
    :return: tensor of model predictions
    """
    model.training = False

    if gpu is not None:
        model = model.cuda(gpu)
        tensor_data = tensor_data.cuda(gpu)

    with torch.no_grad():
        predictions = model(tensor_data).squeeze().cpu()

    return predictions
