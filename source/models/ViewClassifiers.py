import math
import torch
import torch.nn as nn
import numpy as np

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

        n_features = np.prod(self.input_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.n_output_classes),
            nn.Sigmoid(),
            )

    def forward(self, data):
        out = self.classifier(data)
        return out

