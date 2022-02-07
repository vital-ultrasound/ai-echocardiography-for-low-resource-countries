import numpy as np
import torch
import torch.nn as nn


class SimpleVideoClassifier(nn.Module):

    def __init__(self, input_size, n_output_classes=2):
        """
        Simple Video classifier to classify into two classes:

        Args:
            input_size:  shape of the input image. Should be a 2 element vector for a 2D video (width, height).
            n_output_classes: number of output classes
        """

        super().__init__()

        self.name = 'SimpleVideoClassifier'
        self.input_size = input_size
        self.n_output_classes = n_output_classes

        self.n_features = np.prod(self.input_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.n_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.n_output_classes),
            nn.Sigmoid(),
        )

    def forward(self, data):
        out = self.classifier(data)
        return out

def train_loop(dataloader, model_net, loss_fn, optimizer, device):
    """
    train_loop
    Arguments:
        dataloader, model_net, loss_fn, optimizer, device

    Return:
    """
    size = len(dataloader.dataset)
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        clip = data[0]
        label = data[1].to(device)
        out = model_net(clip)
        loss = loss_fn(out, label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            # loss = loss.item()
            loss = loss.detach().item()
            print(f'loss: {loss:>7f}')


def test_loop(dataloader, model_net, loss_fn, device):
    """
    test_loop(dataloader, model, loss_fn):
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            clip = data[0]
            label = data[1].to(device)
            pred = model_net(clip)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")