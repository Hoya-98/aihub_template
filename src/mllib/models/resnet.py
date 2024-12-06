import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from stn import stn

resnet_50 = torchvision.models.resnet.resnet50

resnet_50_weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1


class ResNet(nn.Module):
    def __init__(self,
                 variant: str,
                 pretrained: bool,
                 n_classes: int,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        assert variant in [
            'resnet_50',
        ]

        self.variant = variant
        self.pretrained = pretrained
        self.n_classes = n_classes
        self.stn = stn()

        self.features = None
        self.avgpool = None
        self.fc = None

        self.build()

    def build(self):
        if self.variant == 'resnet_50':
            model = resnet_50(weights=resnet_50_weights)
            channels = 2048

        else:
            raise Exception('unsupported variant')

        self.features = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(in_features=channels, out_features=self.n_classes, bias=True)
        
    def forward(self, x):
        x = self.stn(x)
        z = self.features(x)
        x = self.avgpool(z)
        x = self.flatten(x)
        x = self.fc(x)
        return x, z


if __name__ == '__main__':
    model = ResNet(variant='resnet_50',
                   pretrained=True,
                   n_classes=2)

    print(f"{model.__class__ = }")
    
    print(f"{'input':-^20}")
    x = torch.randn(1,3,224,224)
    print(f"{x.shape = }")

    print(f"{'output':-^20}")
    x, z = model(x)
    print(f"{x.shape = }")
    print(f"{z.shape = }")
    