import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def quantization(x, fixed_bit, decimal_bit):
    mini = -2 ** (fixed_bit - decimal_bit - 1)
    maxi = -mini - 2 ** (-decimal_bit)
    fixedpoint_truevalue = torch.floor(x * 2 ** decimal_bit) / 2 ** decimal_bit
    fixedpoint_truevalue = torch.clip(fixedpoint_truevalue, mini, maxi)
    return fixedpoint_truevalue


class QuantizedConv2d(nn.Module):
    def __init__(self, conv, fixed_bit, decimal_bit):
        super(QuantizedConv2d, self).__init__()
        self.conv = conv
        self.fixed_bit = fixed_bit
        self.decimal_bit = decimal_bit
        # self.b_fl_bits = b_fl_bits

    def freeze(self):
        self.conv.weight = Parameter(
            torch.tensor(quantization(self.conv.weight.data, self.fixed_bit, self.decimal_bit)))

    def quantization_forward(self, x):
        with torch.no_grad():
            x = self.conv(x)
            x = quantization(x, self.fixed_bit, 11)
        return x

