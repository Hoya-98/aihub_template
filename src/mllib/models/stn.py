import torch
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):

    def __init__(
            self,
            n_channel=3,
            img_size=224,
            conv1_kernel_size=7,
            conv2_kernel_size=5,
        ):
        super().__init__()
        
        self.conv1_kernel_size = conv1_kernel_size
        self.conv2_kernel_size = conv2_kernel_size

        self.nx = ((img_size-(self.conv1_kernel_size-1))//2 - (self.conv2_kernel_size-1))//2

        self.localization = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channel, 
                out_channels=8,
                kernel_size=self.conv1_kernel_size
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=8, 
                out_channels=10,
                kernel_size=self.conv2_kernel_size
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),
        )
    
        self.fc_loc = nn.Sequential(
            nn.Linear(10*self.nx**2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3*2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * self.nx * self.nx)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    
    def forward(self, x):   
        x = self.stn(x)
        return x