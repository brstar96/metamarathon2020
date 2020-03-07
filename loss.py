import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable

import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable

def calculate_gradient_penalty(device, batch_size, discriminator, real_images, fake_images, lambda_gp):
    print(device)
    eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.to(device)

    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.to(device)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(
                                  prob_interpolated.size()).to(device),
                              create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return grad_penalty

def l1_loss(input, target):
    """ L1 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(input - target))

##
def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)
