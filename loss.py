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

# def compute_gradient_penalty(device, BATCH_SIZE, Tensor, D, real_samples, fake_samples):
#     """Calculates the gradient penalty loss for WGAN GP"""
#     # Random weight term for interpolation between real and fake samples
#     print('shape of real_samples in loss.py:', real_samples.shape)
#     print('shape of fake_samples in loss.py:', fake_samples.shape)
#
#     alpha = Tensor(np.random.random((BATCH_SIZE, 1, 1, 1)))
#     # alpha = torch.FloatTensor(BATCH_SIZE, 1, 1, 1).uniform_(0, 1)
#     alpha = alpha.expand(BATCH_SIZE, real_samples.size(1), real_samples.size(2), real_samples.size(3))
#     alpha.to(device)
#
#     print('shape of alpha in loss.py:', alpha.shape)
#     # Get random interpolation between real and fake samples
#     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
#     d_interpolates = D(interpolates)
#     fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
#     # Get gradient w.r.t. interpolates
#     gradients = autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=fake,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True,
#     )[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty