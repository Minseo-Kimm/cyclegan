import torch
import torch.nn as nn
import torch.nn.functional as F

def CBR2d(in_chs, out_chs, kernel_size=4, stride=2, padding=1, batchnorm=True, relu=0, path='enc', bias=True):
    # Convolution-Batchnorm-ReLU
    layers = []
    if (path == 'enc'):
        layers += [nn.Conv2d(in_channels=in_chs, out_channels=out_chs, 
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=bias)]
    elif (path == 'dec'):
        layers += [nn.ConvTranspose2d(in_channels=in_chs, out_channels=out_chs, 
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=bias)]

    if (batchnorm):
        layers += [nn.BatchNorm2d(num_features=out_chs)]

    if (relu):
        layers += [nn.LeakyReLU(relu)]
    else:
        layers += [nn.ReLU()]

    cbr = nn.Sequential(*layers)
    return cbr

def CBDR2d(in_chs, out_chs, kernel_size=4, stride=2, padding=1, bias=True):
    # Convolution-Batchnorm-Dropout-ReLU
    layers = []
    layers += [nn.ConvTranspose2d(in_channels=in_chs, out_channels=out_chs, 
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            bias=bias)]
    layers += [nn.BatchNorm2d(num_features=out_chs)]
    layers += [nn.Dropout2d(0.5)]
    layers += [nn.ReLU()]

    cbr = nn.Sequential(*layers)
    return cbr

def C7S1_k(in_chs, out_chs, kernel_size=7, stride=1, padding=3, bias=True):
    layers = []
    layers += [nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            bias=bias, padding_mode='reflect')]
    layers += [nn.InstanceNorm2d(num_features=out_chs)]
    layers += [nn.ReLU()]

    layer = nn.Sequential(*layers)
    return layer

def D_k(in_chs, out_chs, kernel_size=3, stride=2, padding=1, bias=True):
    layers = []
    layers += [nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            bias=bias, padding_mode='reflect')]
    layers += [nn.InstanceNorm2d(num_features=out_chs)]
    layers += [nn.ReLU()]

    layer = nn.Sequential(*layers)
    return layer

def R_k(in_chs, out_chs, kernel_size=3, stride=1, padding=1, bias=True):
    layers = []
    layers += [nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            bias=bias, padding_mode='reflect')]
    layers += [nn.Conv2d(in_channels=out_chs, out_channels=out_chs,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            bias=bias, padding_mode='reflect')]
    
    layer = nn.Sequential(*layers)
    return layer

def U_k(in_chs, out_chs, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True):
    layers = []
    layers += [nn.ConvTranspose2d(in_channels=in_chs, out_channels=out_chs,
                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                    output_padding=output_padding ,bias=bias)]
    layers += [nn.InstanceNorm2d(num_features=out_chs)]
    layers += [nn.ReLU()]

    layer = nn.Sequential(*layers)
    return layer

def C_k(in_chs, out_chs, norm=True, kernel_size=4, stride=2, padding=1, bias=True):
    layers = []
    layers += [nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
    if (norm):
        layers += [nn.InstanceNorm2d(num_features=out_chs)]
    layers += [nn.LeakyReLU(0.2)]

    layer = nn.Sequential(*layers)
    return layer