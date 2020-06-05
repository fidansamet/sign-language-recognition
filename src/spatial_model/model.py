from torch import nn


class BaseModel(nn.Module):

    def __init__(self, in_channels, out_classes):
        super(BaseModel, self).__init__()

        layers_conv = []

        # BLOCK 1
        layers_conv.append(nn.Conv2d(in_channels=in_channels, out_channels=96,
                                kernel_size=7, padding=0, stride=2))
        layers_conv.append(nn.ReLU())
        layers_conv.append(nn.LocalResponseNorm(96))
        layers_conv.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # BLOCK 2
        layers_conv.append(nn.Conv2d(in_channels=96, out_channels=256,
                                kernel_size=5, padding=1, stride=2))
        layers_conv.append(nn.ReLU())
        layers_conv.append(nn.LocalResponseNorm(256))
        layers_conv.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # BLOCK 3
        layers_conv.append(nn.Conv2d(in_channels=256, out_channels=512,
                                kernel_size=3, padding=1, stride=1))
        layers_conv.append(nn.ReLU())

        # BLOCK 4
        layers_conv.append(nn.Conv2d(in_channels=512, out_channels=512,
                                kernel_size=3, padding=1, stride=1))
        layers_conv.append(nn.ReLU())

        # BLOCK 5
        layers_conv.append(nn.Conv2d(in_channels=512, out_channels=512,
                                kernel_size=3, padding=1, stride=1))
        layers_conv.append(nn.ReLU())
        layers_conv.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv_net = nn.Sequential(*layers_conv)

        layers_fc = []

        # BLOCK 6
        layers_fc.append(nn.Linear(25088, 4096))
        layers_fc.append(nn.Dropout2d(0.5))

        # BLOCK 7
        layers_fc.append(nn.Linear(4096, 2048))
        layers_fc.append(nn.Dropout2d(0.5))

        # BLOCK 8
        layers_fc.append(nn.Linear(2048, out_features=out_classes))
        # layers_fc.append(nn.Softmax())

        self.fc_net = nn.Sequential(*layers_fc)
        print(self.conv_net)
        print(self.fc_net)

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(out.size()[0], -1)
        out = self.fc_net(out)
        return out
