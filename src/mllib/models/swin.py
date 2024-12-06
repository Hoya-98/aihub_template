import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from stn import stn

swin_t = torchvision.models.swin_transformer.swin_t
swin_s = torchvision.models.swin_transformer.swin_s
swin_b = torchvision.models.swin_transformer.swin_b

swin_t_weights = torchvision.models.swin_transformer.Swin_T_Weights.IMAGENET1K_V1
swin_s_weights = torchvision.models.swin_transformer.Swin_S_Weights.IMAGENET1K_V1
swin_b_weights = torchvision.models.swin_transformer.Swin_B_Weights.IMAGENET1K_V1


class Swin(nn.Module):
    def __init__(self,
                 variant: str,
                 pretrained: bool,
                 n_classes: int,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        assert variant in [
            'swin_t',
            'swin_s',
            'swin_b',
        ]

        self.variant = variant
        self.pretrained = pretrained
        self.n_classes = n_classes
        self.stn = stn()

        self.build()

    def build(self):
        if self.variant == 'swin_t':
            model = swin_t(weights=swin_t_weights)
            channels = 768

        elif self.variant == 'swin_s':
            model = swin_s(weights=swin_s_weights)
            channels = 768

        elif self.variant == 'swin_b':
            model = swin_b(weights=swin_b_weights)
            channels = 1024
        else:
            raise Exception('unsupported variant')

        self.features = model.features
        self.norm = model.norm
        self.permute = model.permute
        self.avgpool = model.avgpool
        self.flatten = model.flatten
        self.head = nn.Linear(in_features=channels,
                              out_features=self.n_classes,
                              bias=True)
        
    def forward(self, x):
        x = self.stn(x)
        x = self.features(x)
        x = self.norm(x)
        z = self.permute(x)
        x = self.avgpool(z)
        x = self.flatten(x)
        x = self.head(x)
        return x, z


if __name__ == '__main__':
    model = Swin(variant='swin_b',
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
    
    