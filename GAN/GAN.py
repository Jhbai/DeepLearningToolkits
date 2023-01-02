import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__();
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels = nz,
                                                      out_channels = ngf * 8,
                                                      kernel_size = 4,
                                                      stride = 1,
                                                      padding = 0,
                                                      bias = False
                                                     ),
                                   nn.BatchNorm2d(ngf * 8),
                                   nn.ReLU()
                                   )
        ##### (ngf*8) x 4 x 4 #####
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(in_channels = ngf * 8,
                                                      out_channels = ngf * 4,
                                                      kernel_size = 4,
                                                      stride = 2,
                                                      padding = 1,
                                                      bias = False
                                                     ),
                                   nn.BatchNorm2d(ngf * 4),
                                   nn.ReLU()
                                   )
        ##### (ngf*4) x 8 x 8 #####
        self.conv3 = nn.Sequential(nn.ConvTranspose2d(in_channels = ngf * 4,
                                                      out_channels = ngf * 2,
                                                      kernel_size = 4,
                                                      stride = 2,
                                                      padding = 1,
                                                      bias = False
                                                     ),
                                   nn.BatchNorm2d(ngf * 2),
                                   nn.ReLU()
                                   )
        ##### (ngf*2) x 16 x 16 #####
        self.conv4 = nn.Sequential(nn.ConvTranspose2d(in_channels = ngf * 2,
                                                      out_channels = ngf,
                                                      kernel_size = 4,
                                                      stride = 2,
                                                      padding = 1,
                                                      bias = False
                                                     ),
                                   nn.BatchNorm2d(ngf),
                                   nn.ReLU()
                                   )
        ##### ngf x 16 x 16 #####
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(in_channels = ngf,
                                                      out_channels = nc,
                                                      kernel_size = 4,
                                                      stride = 2,
                                                      padding = 1,
                                                      bias = False
                                                     ),
                                   nn.Tanh()
                                   )
    def forward(self, x):
        X = self.conv1(x)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.conv5(X)
        return X
netG = Generator().to(device = 'cuda')
netG.apply(weights_init)