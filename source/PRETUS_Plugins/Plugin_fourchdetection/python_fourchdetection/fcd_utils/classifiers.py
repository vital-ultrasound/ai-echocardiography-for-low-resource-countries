import numpy as np
import torch
import torch.nn as nn


class basicVGG2D_04layers(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, n_frames_per_clip: int = 1):
        """

        References:
        * https://blog.paperspace.com/vgg-from-scratch-pytorch/
        """

        super(basicVGG2D_04layers, self).__init__()
        self.name = 'basicVGG2D_04layers'

        self.in_channels = in_channels
        self.num_classes = num_classes
        #         self.n_frames_per_clip = n_frames_per_clip
        #         self.n_features = np.prod(self.input_size) * self.n_frames_per_clip
        self.flatten = nn.Flatten()

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc0 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(131072, 4096),  # layer1,2,3,4
            ##nn.Linear(7*7*512, 4096), #DEFAULT #
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x.shape(): {x.size()}') ##[batch_size, channels, depth, height, width]
        # x = x.permute(0,2,1,3,4)##[batch_size, depth, channels, height, width]
        # print(f'x.shape(): {x.size()}') #x.shape(): torch.Size([25, 1, 1, 128, 128])
        x = torch.squeeze(x, dim=1)
        # print(f'x.shape(): {x.size()}') #        x.shape(): torch.Size([5, 1, 128, 128])

        x = self.layer1(x)
        # print(f'layer1.size(): {x.size()}')
        x = self.layer2(x)
        # print(f'layer2.size(): {x.size()}')
        x = self.layer3(x)
        # print(f'layer3.size(): {x.size()}')
        x = self.layer4(x)
        # print(f'layer4.size(): {x.size()}')

        x = self.flatten(x)
        # print(f'self.flatten(x) size() {x.size()}')  # x.shape():
        x = self.fc0(x)
        # print(f'self.fc0(x): {x.size()}') #x.shape():
        x = self.fc1(x)
        # print(f'self.fc1(x): {x.size()}') #x.shape():
        x = self.fc2(x)
        # print(f'self.fc2(x): {x.size()}') #x.shape():
        # print(f'x.shape(): {x.size()}')  #x.shape(): torch.Size([4, 2])

        return x


class SimpleVideoClassifier(nn.Module):

    def __init__(self, input_size, n_frames_per_clip, n_classes=2):
        """
        Simple Video classifier to classify into two classes:

        Args:
            input_size:  shape of the input image. Should be a 2 element vector for a 2D video (width, height).
            n_output_classes: number of output classes
        """

        super(SimpleVideoClassifier, self).__init__()
        self.name = 'SimpleVideoClassifier'
        self.input_size = input_size
        self.n_classes = n_classes
        self.n_frames_per_clip = n_frames_per_clip
        self.n_features = np.prod(self.input_size) * self.n_frames_per_clip

        # extract features from each frame
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.n_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_classes),
            # nn.Sigmoid(),
        )

        ##\/ TOREVIEW
        # n_temporal_features = n_spatial_features_out * n_frames
        #
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features=n_temporal_features, out_features=32),
        #     nn.ReLU(),
        #     nn.Linear(in_features=32, out_features=self.n_output_classes),
        #     #nn.Sigmoid(),
        # )
        ##/\ TOREVIEW

    def get_name(self):
        return self.name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x.shape(): {x.size()}') ##[batch_size, channels, depth, height, width]
        # x = x.permute(0,2,1,3,4)##[batch_size, depth,channels, height, width]
        # print(f'x.shape(): {x.size()}')

        x = self.classifier(x)

        return x

    ##\/ TOREVIEW
    # def forward(self, data):
    #
    #     n_frames = data.shape[2]
    #     features = []
    #     for f in range(n_frames):
    #         feat_i = self.frame_features(data[:, :, f, ...])
    #         features.append(feat_i)
    #
    #     feature_vector = torch.stack(features, dim=1)
    #
    #     out = self.classifier(feature_vector)
    #     return out
    ##/\ TOREVIEW
