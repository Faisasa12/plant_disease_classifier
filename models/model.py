from torch import nn
import torch.nn.functional as F


class MyCNN(nn.Module):
  def __init__(self, num_classes=15):
    super().__init__()
    self.conv_block1 = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    self.conv_block2 = nn.Sequential(
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    self.conv_block3 = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    self.conv_block4 = nn.Sequential(
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    self.fc1 = nn.Linear(256 *14 * 14, 512)
    self.fc2 = nn.Linear(512, num_classes)

    self.dropout = nn.Dropout(0.5)




  def forward(self, x):
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.conv_block3(x)
    x = self.conv_block4(x)
    x = x.view(x.shape[0], -1)
    
    x = self.dropout(self.fc1(x))

    x = self.fc2(x)

    x = F.log_softmax(x, dim=1)

    return x