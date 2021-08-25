import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import *

class Generator(nn.Module):
    def __init__(self, in_chs=3, out_chs=3, num_blk=9):
        super(Generator, self).__init__()
        self.num_blk = num_blk

        self.l1 = C7S1_k(in_chs=in_chs, out_chs=64)
        self.l2 = D_k(in_chs=64, out_chs=128)
        self.l3 = D_k(in_chs=128, out_chs=256)
        self.l4 = R_k(in_chs=256, out_chs=256)
        self.l5 = U_k(in_chs=256, out_chs=128)
        self.l6 = U_k(in_chs=128, out_chs=64)
        self.l7 = C7S1_k(in_chs=64, out_chs=out_chs)

        layers = []
        layers += [nn.Tanh()]
        self.final = nn.Sequential(*layers)

    def forward(self, x):

        l1 = self.l1(x)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)
        for _ in range(self.num_blk-1):
            l4 = self.l4(l4)
        l5 = self.l5(l4)
        l6 = self.l6(l5)
        l7 = self.l7(l6)
        output = self.final(l7)

        return output

class Discriminator(nn.Module):
    def __init__(self, in_chs=3, out_chs=1):
        super(Discriminator, self).__init__()

        self.enc1 = C_k(in_chs=in_chs, out_chs=64, norm=False)
        self.enc2 = C_k(in_chs=64, out_chs=128)
        self.enc3 = C_k(in_chs=128, out_chs=256)
        self.enc4 = C_k(in_chs=256, out_chs=512)

        layers = []
        layers += [nn.Conv2d(in_channels=512, out_channels=out_chs,
                                kernel_size=4, stride=1, padding=1, bias=False)]
        self.final = nn.Sequential(*layers)

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        output = self.final(enc4)

        return output

def init_weights(net):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        
    net.apply(init_func)
    return net