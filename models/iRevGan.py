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


class irevnet_block(nn.Module):
  def __init__(self, in_shape1,in_shape2,out_shape1,out_shape2,first=False,affineBN=True,mult=4):
    """ buid invertible bottleneck block """
    super(irevnet_block, self).__init__()
    self.first = first
    self.in_shape1=in_shape1
    self.in_shape2=in_shape2
    self.out_shape1=out_shape1
    self.out_shape2=out_shape2
    assert in_shape1[1]==in_shape1[2]
    assert in_shape2[1]==in_shape2[2]
    assert out_shape1[1]==out_shape1[2]
    assert out_shape2[1]==out_shape2[2]
    assert in_shape1[0]*in_shape1[1]*in_shape1[2]==out_shape2[0]*out_shape2[1]*out_shape2[2]
    assert in_shape2[0]*in_shape2[1]*in_shape2[2]==out_shape1[0]*out_shape1[1]*out_shape1[2]
    if out_shape2[1]==in_shape1[1]*2:
      self.psi1 = psi(2,1)
    elif in_shape1[1]==out_shape2[1]*2:
      self.psi1 = psi(2,-1)
    else:
      self.psi1 = psi(2,0)
    if out_shape1[1]==in_shape2[1]*2:
      self.psi2 = psi(2,1)
    elif in_shape2[1]==out_shape1[1]*2:
      self.psi2 = psi(2,-1)
    else:
      self.psi2 = psi(2,0)
    
    layers = []

    if in_shape1[1] > 1:
      if not first:
        layers.append(nn.BatchNorm2d(in_shape1[0], affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
    if out_shape2[1]==in_shape2[1]*2:
      layers.append(nn.ConvTranspose2d(in_shape2[0], int(in_shape2[0]//mult), kernel_size=3,
                  stride=2, padding=1,output_padding =1, bias=False))
    elif 2*out_shape2[1]==in_shape2[1]:
      layers.append(nn.Conv2d(in_shape2[0], int(in_shape2[0]//mult), kernel_size=3,
                  stride=2, padding=1, bias=False))
    else:
      layers.append(nn.Conv2d(in_shape2[0], int(in_shape2[0]//mult), kernel_size=3,
                  stride=1, padding=1, bias=False))
    #layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
    #layers.append(nn.ReLU(inplace=True))
    #layers.append(nn.Conv2d(int(out_ch//mult), int(out_ch//mult),
    #             kernel_size=3, padding=1, bias=False))
    #layers.append(nn.Dropout(p=dropout_rate))
    if in_shape1[1] > 1:
      layers.append(nn.BatchNorm2d(int(in_shape2[0]//mult), affine=affineBN))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(int(in_shape2[0]//mult), out_shape2[0], kernel_size=3,
                  padding=1, bias=False))
    self.bottleneck_block = nn.Sequential(*layers)

  def forward(self, x):
    """ bijective or injective block forward """
    x1,x2 = x[0],x[1]
    Fx2 = self.bottleneck_block(x2)
    x1 = self.psi1.forward(x1)
    x2 = self.psi2.forward(x2)
    y1 = Fx2 + x1
    return (x2, y1)

  def inverse(self, x):
    """ bijective or injecitve block inverse """
    x2, y1 = x[0], x[1]
    x2 = self.psi2.inverse(x2)
    Fx2 = - self.bottleneck_block(x2)
    x1 = Fx2 + y1
    x1 = self.psi1.inverse(x1)
    return (x1, x2)

class resnet_block(nn.Module):
  def __init__(self, in_ch, out_ch, stride=1, first=False,affineBN=True,mult=4):
    """ buid invertible bottleneck block """
    super(resnet_block, self).__init__()
    self.first = first
    self.stride = stride
    layers = []
    in_ch=in_ch
    out_ch=out_ch
    self.psi = psi(stride,-1)
    self.injective_pad = injective_pad(out_ch-in_ch*stride*stride)
    if not first:
      layers.append(nn.BatchNorm2d(in_ch, affine=affineBN))
      layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(in_ch, int(out_ch//mult), kernel_size=3,
                  stride=stride, padding=1, bias=False))
    #layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
    #layers.append(nn.ReLU(inplace=True))
    #layers.append(nn.Conv2d(int(out_ch//mult), int(out_ch//mult),
    #             kernel_size=3, padding=1, bias=False))
    #layers.append(nn.Dropout(p=dropout_rate))
    layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(int(out_ch//mult), out_ch, kernel_size=3,
                  padding=1, bias=False))
    self.bottleneck_block = nn.Sequential(*layers)

  def forward(self, x):
    
    Fx = self.bottleneck_block(x)
    if self.stride == 2:
      x = self.injective_pad(self.psi.inverse(x))
    y = Fx + x
    return y


class iRevGener(nn.Module):
  def __init__(self,output_shape,block_num=1,mult=2):
    super(iRevGener, self).__init__()
    self.output_shape = output_shape # [C,H,W]
    self.stride_number = math.ceil(math.log(output_shape[1],2))
    self.output_shape_raw = output_shape
    self.output_shape_raw[1] = 2**self.stride_number
    self.output_shape_raw[2] = 2**self.stride_number
    self.input_shape_raw =  self.output_shape_raw[0]*self.output_shape_raw[1]*self. output_shape_raw[2]
    self.first = True
    self.lastpsi = psi(2)
    nChannels = []
    for i in range(self.stride_number-1):
      nChannels = nChannels + [self.input_shape_raw//(4**i)]
    nBlocks = [block_num]*(self.stride_number-1)
    nStrides = [2]*(self.stride_number-1)
    self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,nStrides,
                                    in_ch=self.input_shape_raw, mult=mult)
    
    
  def irevnet_stack(self, _block, nChannels, nBlocks, nStrides , in_ch, mult):
    """ Create stack of irevnet blocks """
    block_list = nn.ModuleList()
    strides = []
    channels = []
    in_sizes = [1]
    for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
      strides = strides + ([stride] + [1]*(depth-1))
      channels = channels + ([channel//(stride**2)]*(depth))
    for stride in strides:
      in_sizes = in_sizes + [in_sizes[-1]*stride]
    for channel, in_size, stride in zip(channels, in_sizes, strides):
      in_shape1 = (int(in_ch//2),in_size,in_size)
      in_shape2 = in_shape1
      out_shape1 = (int(in_ch//(2*stride**2)),in_size*stride,in_size*stride)
      out_shape2 = out_shape1
      block_list.append(_block(in_shape1,in_shape2,out_shape1,out_shape2,first=self.first,mult=mult))
      in_ch = channel
      self.first = False
    return block_list

  def forward(self, x):
    """ irevnet forward """
    n = self.input_shape_raw//2
    out = (x[:, :n, :, :], x[:, n:, :, :])
    for block in self.stack:
      out = block.forward(out)
    out = merge(out[0], out[1])
    
    self.lastpsi.forward(out)
    return out

  def inverse(self, out):
    """ irevnet inverse """
    out = self.lastpsi.inverse(out)
    out = split(out)
    for i in range(len(self.stack)):
        out = self.stack[-1-i].inverse(out)
    out = merge(out[0],out[1])
    return out
    
class Disc(nn.Module):
  def __init__(self,input_shape,block_num=1,nClasses = 1,mult=2):
    super(Disc, self).__init__()
    self.input_shape = input_shape # [C,H,W]
    self.nClasses = nClasses
    self.stride_number = math.ceil(math.log(input_shape[1],2))
    #self.input_shape_raw = input_shape
    #self.input_shape_raw[1] = 2**self.stride_number
    #self.input_shape_raw[2] = 2**self.stride_number
    #self.input_shape_raw =  self.output_shape_raw[0]*self.output_shape_raw[1]*self. output_shape_raw[2]
    self.first = True
    #self.lastpsi = psi(2)
    nChannels = [64]
    for i in range(self.stride_number-1):
      nChannels = nChannels + [nChannels[-1]*2]
    nBlocks = [block_num]*self.stride_number
    nStrides = [2]*self.stride_number
    self.stack = self.irevnet_stack(resnet_block, nChannels, nBlocks,nStrides,
                                    in_ch=self.input_shape[0], mult=mult)
    self.linear = nn.Linear(nChannels[-1], self.nClasses)
    
  def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, in_ch, mult):
    """ Create stack of irevnet blocks """
    block_list = nn.ModuleList()
    strides = []
    channels = []
    for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
      strides = strides + ([stride] + [1]*(depth-1))
      channels = channels + ([channel]*depth)
    for channel, stride in zip(channels, strides):
      block_list.append(_block(in_ch, channel, stride,first=self.first,mult=mult))
      in_ch = channel
      self.first = False
    return block_list

  def forward(self, x):
    """ irevnet forward """
    out = x
    for block in self.stack:
      out = block.forward(out)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out