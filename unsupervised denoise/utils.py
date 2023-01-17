import math
from torch import Tensor, cat, zeros, ones,  randperm, bernoulli, full


# percent:percent of zeros in mask
def create_mask(channels=3, height=128, width=128, percent=0.3, probability=0.5, mode='percent'):
    if mode == 'percent':
        num = width*height
        num_zeros = math.floor(num*percent)
        num_ones = num-num_zeros
        x = cat((zeros(num_zeros), ones(num_ones)))
        x = x[randperm(num)]
        x = x.view((height, width)).unsqueeze(
            0).expand((channels, height, width))
        return x
    elif mode == 'bernoulli':
        return bernoulli(full((height, width), probability)).expand((channels, height, width))
