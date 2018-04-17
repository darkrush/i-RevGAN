"""
Code for "i-RevNet: Deep Invertible Networks"
https://openreview.net/pdf?id=HJsjkMb0Z
ICLR, 2018

(c) Joern-Henrik Jacobsen, 2018
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .model_utils import split, merge, injective_pad, psi

import torch.nn.parallel


class WGAN_D(nn.Module):
  def __init__(self, isize, nc, ndf, n_extra_layers=2, if_BN = True):
    super(WGAN_D, self).__init__()
    assert isize % 16 == 0, "isize has to be a multiple of 16"

    main = nn.Sequential()
    # input is nc x isize x isize
    main.add_module('initial.conv.{0}-{1}'.format(nc, ndf),
                    nn.Conv2d(nc, ndf, 3, 2, 1, bias=False))
    main.add_module('initial.relu.{0}'.format(ndf),
                    nn.LeakyReLU(0.2, inplace=True))
    csize, cndf = isize / 2, ndf

    # Extra layers
    for t in range(n_extra_layers):
      main.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                      nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
      if if_BN :
        main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cndf),
                        nn.BatchNorm2d(cndf))
      main.add_module('extra-layers-{0}.{1}.relu'.format(t, cndf),
                      nn.LeakyReLU(0.2, inplace=True))

    while csize > 2:
      in_feat = cndf
      out_feat = cndf * 2
      main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                      nn.Conv2d(in_feat, out_feat, 3, 2, 1, bias=False))
      if if_BN :
        main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                        nn.BatchNorm2d(out_feat))
      main.add_module('pyramid.{0}.relu'.format(out_feat),
                      nn.LeakyReLU(0.2, inplace=True))
      cndf = cndf * 2
      csize = csize / 2
      
      
    self.finalsize = int(csize*csize*cndf)
    self.main = main
    self.output = nn.Linear(self.finalsize, 1,bias=False)

  def forward(self, input):
    output = self.main(input)
    output = output.view(-1,self.finalsize)
    output = self.output(output)
    return output


class WGAN_G(nn.Module):
  def __init__(self, insize =128 ):
    super(WGAN_G, self).__init__()
    
    

    self.first_linear = nn.Linear(insize, 8*8*128,bias=True)
    layers = []

    layers.append(nn.Upsample(scale_factor = 2 ,mode='bilinear'))
    layers.append(nn.Conv2d(128, 64, kernel_size=3,stride=1, padding=1, bias=True))
    #layers.append(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1,output_padding =1, bias=True))
    layers.append(nn.BatchNorm2d(64, affine=True))
    layers.append(nn.ReLU(inplace=True))
    
    layers.append(nn.Conv2d(64, 32, kernel_size=3,stride=1, padding=1, bias=True))
    layers.append(nn.BatchNorm2d(32, affine=True))
    layers.append(nn.ReLU(inplace=True))

    layers.append(nn.Upsample(scale_factor = 2,mode='bilinear'))
    layers.append(nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=True))
    #layers.append(nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1,output_padding =1, bias=True))
    layers.append(nn.BatchNorm2d(16, affine=True))
    layers.append(nn.ReLU(inplace=True))
    
    layers.append(nn.Conv2d(16, 1, kernel_size=3,stride=1, padding=1, bias=False))
    layers.append(nn.Sigmoid())
    self.net_OP = nn.Sequential(*layers)

  def forward(self, input):
    output = input.view(-1,128)
    output = self.first_linear(output)
    output = output.view(-1,128,8,8)
    output = self.net_OP(output)
    return output