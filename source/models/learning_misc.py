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

    def forward(self, data):
        out = self.classifier(data)
        return out


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x #[10, 3, 3, 3]


def train_validation_loop(train_dataloader, validation_dataloader, model, criterion, optimizer, device):
    """
    train_loop
    Arguments:
        dataloader, model, criterion, optimizer, device

    Return:
    """
    train_epoch_loss = 0
    train_epoch_acc = 0
    step_train = 0
    step_val = 0
    for clip_batch_idx, sample_batched in enumerate(train_dataloader):
        step_train += 1
        print(f' BATCH_OF_CLIPS_INDEX: {clip_batch_idx} ')
        X_train_batch, y_train_batch = sample_batched[0].to(device), sample_batched[1].to(device)
        optimizer.zero_grad()
        # print(f'----------------------------------------------------------')
        # print(f'   X_train_batch.size(): {X_train_batch.size()}') # torch.Size([9, 60, 1, 128, 128]) clips, frames, channels, [width, height]
        # print(f'   y_train_batch.size(): {y_train_batch.size()}') # torch.Size([9])

        #y_train_pred = model(X_train_batch) #torch.Size([9, 2])
        y_train_pred = model(X_train_batch).squeeze() #torch.Size([9, 2])
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = binary_accuracy(y_train_pred, y_train_batch)
        #print(train_loss)

        # Backpropagation
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.detach().item()
        train_epoch_acc += train_acc.detach().item()
        #print(f'train_loss: {train_epoch_loss:.4f}')

        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_epoch_acc = 0
            for clip_batch_idx, sample_val_batched in enumerate(validation_dataloader):
                step_val += 1
                X_val_batch, y_val_batch = sample_val_batched[0].to(device), sample_val_batched[1].to(device)
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch).squeeze()
                #y_val_pred = torch.unsqueeze(y_val_pred, 0)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = binary_accuracy(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.detach().item()
                val_epoch_acc += val_acc.detach().item()

    train_epoch_loss /= step_train
    train_epoch_acc /= step_train
    val_epoch_loss /= step_val
    val_epoch_acc /= step_val

    return train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc


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