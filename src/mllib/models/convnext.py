import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from stn import stn

convnext_t = torchvision.models.convnext.convnext_tiny
convnext_s = torchvision.models.convnext.convnext_small
convnext_b = torchvision.models.convnext.convnext_base

convnext_t_weights = torchvision.models.convnext.ConvNeXt_Tiny_Weights .IMAGENET1K_V1
convnext_s_weights = torchvision.models.convnext.ConvNeXt_Small_Weights.IMAGENET1K_V1
convnext_b_weights = torchvision.models.convnext.ConvNeXt_Base_Weights .IMAGENET1K_V1


class ConvNeXt(nn.Module):
    def __init__(self,
                 variant: str,
                 pretrained: bool,
                 n_classes: int,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        assert variant in [
            'convnext_t',
            'convnext_s',
            'convnext_b',
        ]

        self.variant = variant
        self.pretrained = pretrained
        self.n_classes = n_classes
        self.stn = stn()

        self.features = None
        self.avgpool = None
        self.classifier = None

        self.build()

    def build(self):
        if self.variant == 'convnext_t':
            model = convnext_t(weights=convnext_t_weights)
            channels = 768

        elif self.variant == 'convnext_s':
            model = convnext_s(weights=convnext_s_weights)
            channels = 768

        elif self.variant == 'convnext_b':
            model = convnext_b(weights=convnext_b_weights)
            channels = 1024

        else:
            raise Exception('unsupported variant')

        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        self.classifier[2] = nn.Linear(in_features=channels,
                                       out_features=self.n_classes,
                                       bias=True)
        
    def forward(self, x):
        x = self.stn(x)
        z = self.features(x)
        x = self.avgpool(z)
        x = self.classifier(x)
        return x, z


if __name__ == '__main__':
    model = ConvNeXt(variant='convnext_t',
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
    
    