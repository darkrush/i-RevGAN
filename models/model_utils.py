"""
Code for "i-RevNet: Deep Invertible Networks"
https://openreview.net/pdf?id=HJsjkMb0Z
ICLR 2018
"""

import torch
import torch.nn as nn

from torch.nn import Parameter


def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :self.pad_size, :, :]


class psi(nn.Module):
    def __init__(self, stride=1, if_upsample=0):
        super(psi, self).__init__()
        self.stride = stride
        self.if_upsample = if_upsample
    def down_sample(self, input):
        downscale_factor = self.stride
        batch_size, channels, in_height, in_width = input.size()
        out_height = int(in_height / downscale_factor)
        out_width = int(in_width / downscale_factor)
        input_view = input.contiguous().view(batch_size, channels, out_height, downscale_factor,out_width, downscale_factor)
        channels = channels*(downscale_factor ** 2)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return shuffle_out.view(batch_size, channels, out_height, out_width)
    def up_sample(self, input):
        output = nn.functional.pixel_shuffle(input,self.stride)
        return output.contiguous()
    def forward(self, input):
        if self.stride == 1:
            return input    
        elif self.if_upsample == 1:
            return self.up_sample(input)
        elif self.if_upsample == -1:
            return self.down_sample(input)
        else:
            return input
    def inverse(self, input):
        if self.stride == 1:
            return input
        elif self.if_upsample == 1:
            return self.down_sample(input)    
        elif self.if_upsample == -1:
            return self.up_sample(input)
        else:
            return input
    

class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


def get_all_params(var, all_params):
    if isinstance(var, Parameter):
        all_params[id(var)] = var.nelement()
    elif hasattr(var, "creator") and var.creator is not None:
        if var.creator.previous_functions is not None:
            for j in var.creator.previous_functions:
                get_all_params(j[0], all_params)
    elif hasattr(var, "previous_functions"):
        for j in var.previous_functions:
            get_all_params(j[0], all_params)
