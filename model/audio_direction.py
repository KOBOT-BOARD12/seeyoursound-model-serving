import torch.nn as nn
import torch.nn.functional as F


class SoundDirection(nn.Module):
    def __init__(self):
        super(SoundDirection, self).__init__()
        self.CNNLayer = nn.Sequential(
            nn.Conv2d(1, 64, (2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 64, (2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (2, 2), stride=(2, 2)),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.LinearLayer = nn.Sequential(
            nn.Linear(12288, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 3),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        conv_output = self.CNNLayer(x)
        flatten = self.flatten(conv_output)
        linear_output = self.LinearLayer(flatten)

        return linear_output
