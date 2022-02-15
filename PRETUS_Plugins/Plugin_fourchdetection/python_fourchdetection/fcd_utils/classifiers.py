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

        n_features = np.prod(self.input_size)
        n_frames = self.input_size[1]
        n_spatial_features = np.prod(self.input_size[-2:])
        n_spatial_features_out = 32

        # extract features from each frame
        self.frame_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_spatial_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_spatial_features_out),
            nn.ReLU()
        )

        n_temporal_features = n_spatial_features_out * n_frames

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_temporal_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.n_output_classes),
            #nn.Sigmoid(),
        )

    def get_name(self):
        return self.name

    def forward(self, data):

        n_frames = data.shape[2]
        features = []
        for f in range(n_frames):
            feat_i = self.frame_features(data[:, :, f, ...])
            features.append(feat_i)

        feature_vector = torch.stack(features, dim=1)

        out = self.classifier(feature_vector)
        return out
