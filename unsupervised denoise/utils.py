import math
from torch import tensor, cat, zeros, ones,  randperm, bernoulli, full
from torch.nn.functional import conv2d


# percent:percent of zeros in mask
def create_mask(channels=3, height=512, width=512, percent=0.3, probability=0.2, mode='bernoulli', device='cuda'):
    if mode == 'percent':
        num = width*height*channels
        num_zeros = math.floor(num*percent)
        num_ones = num-num_zeros
        x = cat((zeros(num_zeros, device=device), ones(num_ones, device=device)))
        x = x[randperm(num)]
        x = x.view((channels, height, width))
        return x
    elif mode == 'bernoulli':
        return bernoulli(full((channels, height, width), 1-probability, device=device))


def image_gradient(img, mode='sobel', device='cuda'):
    if mode == 'weak':
        kernel_x = tensor(data=[[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]],
                          device=device).repeat(3, 3, 1, 1)
        kernel_y = tensor(data=[[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]],
                          device=device).repeat(3, 3, 1, 1)
    elif mode == 'sobel':
        kernel_x = tensor(data=[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                          device=device).repeat(3, 3, 1, 1)
        kernel_y = tensor(data=[[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]],
                          device=device).repeat(3, 3, 1, 1)
    dx = conv2d(img, weight=kernel_x, padding=1)
    dy = conv2d(img, weight=kernel_y, padding=1)
    return 0.5*(dx+dy)
