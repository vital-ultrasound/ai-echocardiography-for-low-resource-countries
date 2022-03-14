import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNNClassifier(nn.Module):

    def __init__(self, input_size, n_classes=2):
        """
        Simple Video classifier to classify into two classes:

        Args:
            input_size:  shape of the input image. Should be a 2 element vector for a 2D video (width, height) [e.g. 128, 128].
            n_classes: number of output classes
        """

        super(BasicCNNClassifier, self).__init__()
        self.name = 'BasicCNNClassifier'
        self.input_size = input_size
        self.n_classes = n_classes
        self.n_frames_per_clip = 60
        self.n_features = np.prod(self.input_size)*self.n_frames_per_clip

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.n_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        #print(f'  x.size():::::::  {x.size()}')  #   x.size():::::::  torch.Size([2, 2])
        #print(x)
        #tensor([[0.1271, 0.6632],
        #        [0.3063, 0.5489]], device='cuda:0', grad_fn= < SigmoidBackward >)
        return x

class basicVGGNet(nn.Module):

    def __init__(self, tensor_shape_size, n_classes=2, cnn_channels=(1, 16, 32)):
        """
        Simple Visual Geometry Group Network (VGGNet) to classify two US image classes (background and 4CV).

        Args:
            tensor_shape_size: [Batch_clips, Depth, Channels, Height, Depth]

        """
        super(basicVGGNet, self).__init__()
        self.name = 'basicVGGNet'

        self.tensor_shape_size = tensor_shape_size
        self.n_classes = n_classes

        # define the CNN
        self.n_output_channels = cnn_channels ##  self.n_output_channels::: (1, 16, 32)
        self.kernel_size = (3, ) * (len(cnn_channels) -1) ## self.kernel_size::: (3, 3)

        self.n_batch_size_of_clip_numbers = self.tensor_shape_size[0]
        self.n_frames_per_clip = self.tensor_shape_size[1]
        self.n_number_of_image_channels = self.tensor_shape_size[2]
        self.input_shape_tensor = self.n_batch_size_of_clip_numbers * self.n_frames_per_clip * self.n_number_of_image_channels

        self.conv1 = nn.Conv3d(in_channels=self.n_number_of_image_channels, out_channels=64,
                               kernel_size=(1, 1, 1), stride=(1, 1, 1), padding = (0, 0, 0)
                               )
                    #IN: [N,Cin,D,H,W]; OUT: (N,Cout,Dout,Hout,Wout)
                    #[batch_size, channels, depth, height, width].

        self.maxpool3d = nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.fc1 = nn.Linear(in_features=62914560, out_features=self.n_classes)

    def forward(self, x):
        print(f'x.shape(): {x.size()}') #x.shape(): torch.Size([10, 60, 1, 128, 128])
        x = torch.permute(x, (0, 2, 1 , 3, 4)) ##[batch_size, channels, depth, height, width]
        print(f'x.shape(): {x.size()}') #x.shape(): torch.Size([10, 1, 60, 128, 128])
        # x = F.relu(self.conv1(x))
        # x = self.maxpool3d(x)
        # x = x.reshape(x.shape[0], -1)
        # x = F.dropout(x, p=0.5) #dropout was included to combat overfitting
        # x = self.fc1(x)

        return x

def train_loop(train_dataloader, model, criterion, optimizer, device):
    """
    train_loop
    Arguments:
        dataloader, model, criterion, optimizer, device

    Return:
    """
    train_epoch_loss = 0
    step_train = 0
    #size = len(train_dataloader.dataset)
    for clip_batch_idx, sample_batched in enumerate(train_dataloader):
        step_train += 1
        X_train_batch, y_train_batch = sample_batched[0].to(device), sample_batched[1].to(device)

        print(f' BATCH_OF_CLIPS_INDEX: {clip_batch_idx} ')
        # print(f'----------------------------------------------------------')
        # print(f'   X_train_batch.size(): {X_train_batch.size()}') # torch.Size([9, 60, 1, 128, 128]) clips, frames, channels, [width, height]
        # print(f'   y_train_batch.size(): {y_train_batch.size()}') # torch.Size([9])

        # Compute prediction and loss
        # y_train_pred = model(X_train_batch) #torch.Size([9, 2])
        y_train_pred = model(X_train_batch).squeeze()  # torch.Size([9, 2])
        train_loss = criterion(y_train_pred, y_train_batch)
        #train_acc = binary_accuracy(y_train_pred, y_train_batch)
        # print(train_loss)

        # Backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # if clip_batch_idx % 10 == 0: ## Print loss values every 10 clip batches
        #     train_loss, current = train_loss.item(), clip_batch_idx * len(X_train_batch)
        #     print(f"loss: {train_loss:>7f}  [{current:>5d}/{size:>5d}]")

        train_epoch_loss += train_loss.detach().item()


    train_epoch_loss /= step_train

    return train_epoch_loss

def binary_accuracy(y_pred, y_test):
    """
    binary_accuracy to calculate accuracy per epoch.
    """
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    accuracy = correct_results_sum/y_test.shape[0]
    accuracy = torch.round(accuracy * 100)
    return accuracy


def test_loop(dataloader, model, criterion, device):
    """
    test_loop(dataloader, model, criterion, device)
    """

    train_epoch_acc = 0
    step_test = 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_epoch_loss, correct = 0, 0

    with torch.no_grad():
        #model.eval()
        #val_epoch_loss = 0
        #val_epoch_acc = 0
        for clip_batch_idx, sample_val_batched in enumerate(dataloader):
            step_test += 1
            X_val_batch, y_val_batch = sample_val_batched[0].to(device), sample_val_batched[1].to(device)
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

            y_val_pred = model(X_val_batch)
            test_epoch_loss += criterion(y_val_pred, y_val_batch).detach().item()
            correct += (y_val_pred.argmax(1) == y_val_batch).type(torch.float).sum().detach().item()

    test_epoch_loss /= num_batches
    correct /= size

    return test_epoch_loss, correct