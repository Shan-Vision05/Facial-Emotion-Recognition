from torch import nn

class FER2013_Model(nn.Module):
  def __init__(self, input_shape, output_shape):
    super().__init__()

    # input shape [32, 1, 48, 48]
    self.ConvBlock1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )
    # output shape [64, 1, 46, 46]

    #--------------------------------------------

    # input shape [64, 1, 46, 46]
    self.ConvBlock2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
        nn.BatchNorm2d(128),
        nn.ReLU(),
    )
    # output shape [128, 1, 44, 44]

    #--------------------------------------------

    # input shape [128, 1, 44, 44]
    self.ConvBlock3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )
    # output shape [64, 1, 21, 21]

    #--------------------------------------------

    # input shape [64, 1, 21, 21]
    self.ConvBlock4 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
    )
    # output shape [32, 1, 19, 19]

    #--------------------------------------------

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=32*19*19, out_features=2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Linear(in_features=2048, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=output_shape),

    )

  def forward(self, x):
    x = self.ConvBlock1(x)
    x = self.ConvBlock2(x)
    x = self.ConvBlock3(x)
    x = self.ConvBlock4(x)

    x = self.classifier(x)

    return x