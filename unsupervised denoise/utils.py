import math
from torch import Tensor, cat, zeros, ones,  randperm, bernoulli, full
from zmq import device


# percent:percent of zeros in mask
def create_mask(channels=3, height=128, width=128, percent=0.3, probability=0.5, mode='percent', device='cpu'):
    if mode == 'percent':
        num = width*height
        num_zeros = math.floor(num*percent)
        num_ones = num-num_zeros
        x = cat((zeros(num_zeros, device=device), ones(
            num_ones, device=device)))
        x = x[randperm(num)]
        x = x.view((height, width)).unsqueeze(
            0).expand((channels, height, width))
        return x
    elif mode == 'bernoulli':
        return bernoulli(full((height, width), probability, device=device)).expand((channels, height, width))
