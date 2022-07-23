import torch.nn as nn
import torch
from torchinfo import summary


class Segment_character(nn.Module):
    def __init__(self, num_class):
        super(Segment_character, self).__init__()
        self.num_class = num_class

        self.layers = [nn.Conv2d(1, 16, kernel_size=(22, 22), padding='same'), nn.BatchNorm2d(16), nn.ReLU(),
                       nn.Conv2d(16, 32, kernel_size=(16, 16), padding='same'), nn.BatchNorm2d(32), nn.ReLU(),
                       nn.Conv2d(32, 64, kernel_size=(8, 8), padding='same'), nn.BatchNorm2d(64), nn.ReLU(),
                       nn.Conv2d(64, 64, kernel_size=(4, 4), padding='same'), nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=(4, 4)),
                       nn.Flatten(),
                       nn.Linear(3136, 128), nn.Dropout(0.5), nn.ReLU(),
                       nn.Linear(128, num_class)]
        self.sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.sequential(x)
        return x


if __name__ == '__main__':
    model = Segment_character(36)
    x = torch.randn(2, 1, 28, 28)
    output = model(x)
    summary(model, (1, 1, 28, 28))
