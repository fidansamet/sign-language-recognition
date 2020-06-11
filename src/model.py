from torch import nn
import torchvision.models as models
import config as cfg
import torch
import os


class BaseModel(nn.Module):

    def __init__(self, in_channels, out_classes, flatten_size, fuse_early=0):
        super(BaseModel, self).__init__()

        layers_conv = []

        # BLOCK 1
        layers_conv.append(nn.Conv2d(in_channels=in_channels, out_channels=96,
                                     kernel_size=7, padding=0, stride=2))
        layers_conv.append(nn.ReLU(inplace=True))
        layers_conv.append(nn.BatchNorm2d(96))
        layers_conv.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # BLOCK 2
        layers_conv.append(nn.Conv2d(in_channels=96, out_channels=256,
                                     kernel_size=5, padding=1, stride=2))
        layers_conv.append(nn.ReLU(inplace=True))
        layers_conv.append(nn.BatchNorm2d(256))
        layers_conv.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # BLOCK 3
        layers_conv.append(nn.Conv2d(in_channels=256, out_channels=512,
                                     kernel_size=3, padding=1, stride=1))
        layers_conv.append(nn.ReLU(inplace=True))

        # BLOCK 4
        layers_conv.append(nn.Conv2d(in_channels=512, out_channels=512,
                                     kernel_size=3, padding=1, stride=1))
        layers_conv.append(nn.ReLU(inplace=True))

        # BLOCK 5
        layers_conv.append(nn.Conv2d(in_channels=512, out_channels=512,
                                     kernel_size=3, padding=1, stride=1))
        layers_conv.append(nn.ReLU(inplace=True))
        layers_conv.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv_net = nn.Sequential(*layers_conv)

        if fuse_early:
            self.classifier = nn.Sequential(
                # nn.Linear(512 * 7 * 7, 4096),
                # nn.Linear(32768, 4096),
                nn.Linear(flatten_size, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 2048),
                nn.ReLU(True),
                nn.Dropout(),
            )
        else:
            self.classifier = nn.Sequential(
                # nn.Linear(512 * 7 * 7, 4096),
                # nn.Linear(32768, 4096),
                nn.Linear(flatten_size, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 2048),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(2048, out_classes),
            )

        # layers_fc = []
        #
        # # BLOCK 6
        # layers_fc.append(nn.Linear(25088, 4096))
        # layers_fc.append(nn.Dropout2d(0.5))
        #
        # # BLOCK 7
        # layers_fc.append(nn.Linear(4096, 2048))
        # layers_fc.append(nn.Dropout2d(0.5))
        #
        # # BLOCK 8
        # layers_fc.append(nn.Linear(2048, out_features=out_classes))
        # # layers_fc.append(nn.Softmax())
        #
        # self.fc_net = nn.Sequential(*layers_fc)

        # init weights
        self._initialize_weights()

        print(self.conv_net)
        print(self.classifier)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(out.size()[0], -1)
        out = self.classifier(out)
        return out


class FinalFcLayer(nn.Module):
    def __init__(self, out_classes):
        super(FinalFcLayer, self).__init__()
        self.final_fc = nn.Linear(4096, out_classes)
        self._initialize_weights()
        print(self.final_fc)

    def forward(self, rgb_out, flow_out):
        out = torch.cat((rgb_out, flow_out), dim=1)
        out = self.final_fc(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class FusedModel(nn.Module):
    def __init__(self, fuse_type=0, pretrained=1):
        super(FusedModel, self).__init__()

        self.spatial_model = BaseModel(cfg.SPATIAL_IN_CHANNEL, len(cfg.CLASSES), cfg.SPATIAL_FLATTEN, fuse_early=fuse_type)
        self.temporal_model = BaseModel(cfg.TEMPORAL_IN_CHANNEL, len(cfg.CLASSES), cfg.TEMPORAL_FLATTEN, fuse_early=fuse_type)

        if pretrained:
            path_spatial = os.path.join(cfg.INIT_MODEL_PATH + 'spatial_1', 'spatial_model-100.pkl')
            self.spatial_model.load_state_dict(torch.load(path_spatial), strict=False)

            path_temporal = os.path.join(cfg.INIT_MODEL_PATH + 'temporal_1', 'spatial_model-5.pkl')
            self.temporal_model.load_state_dict(torch.load(path_temporal), strict=False)



        if fuse_type: # for early fusion
            self.fuse = FinalFcLayer(len(cfg.CLASSES))

         ###### asagisi comment

        # if fuse_type == cfg.LATE:
        #     self.spatial_model = BaseModel(cfg.SPATIAL_IN_CHANNEL, len(cfg.CLASSES), 25088)
        #     self.temporal_model = BaseModel(cfg.TEMPORAL_IN_CHANNEL, len(cfg.CLASSES), 32768)
        #     if pretrained:
        #         path_spatial = os.path.join(cfg.INIT_MODEL_PATH + 'spatial_1', 'spatial_model-100.pkl')
        #         self.spatial_model.load_state_dict(path_spatial, strict=False)
        #
        #
        #         self.spatial_model = load_model('spatial', 100)
        #         self.temporal_model = load_model('temporal', 5)
        #
        # else:
        #     if scratch:
        #         self.spatial_model = BaseModel(cfg.SPATIAL_IN_CHANNEL, len(cfg.CLASSES), 25088, fuse_early=1)
        #         self.temporal_model = BaseModel(cfg.TEMPORAL_IN_CHANNEL, len(cfg.CLASSES), 32768, fuse_early=1)
        #         self.fuse = FinalFcLayer(len(cfg.CLASSES))
        #     else:
        #         self.spatial_model = load_model('spatial', 100)
        #         self.temporal_model = load_model('temporal', 5)
        #         self.fuse = FinalFcLayer(len(cfg.CLASSES))
        #########


