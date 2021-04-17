# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch


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


a = torch.randn((15, 1))
b = torch.randn((9, 1))

torch.vstack((a, b)).shape
