import os
import time
import torch
import datetime
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from sagan_models import Generator, Discriminator
from utils import *

import torch


def fft2(data, real=False):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        :param data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -2 & -1 are spatial dimensions and dimension -3 has size 1 or 2. All other dimensions are
            assumed to be batch dimensions -> Typically size B x 1 or 2 x H x W
        :param real
    Returns:
        :return torch.Tensor: The 2D FFT of the input, with dimension B x 2 x H x W
    """
    data = ifftshift(data, dim=(-2, -1))
    if not real:
        assert data.size(-3) == 2
        data = torch.fft(ctoend(data), 2, normalized=True)
    else:
        assert data.size(-3) == 1
        data = torch.rfft(data, 2, normalized=True, onesided=False).squeeze(-4)
    data = fftshift(data, dim=(-3, -2))
    return cto1(data)


def ifft2(data, real=False):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        :param data (torch.Tensor): Complex valued input data with at least 3 dimensions, typically of dimension
            B x 2 x H x W
        :param real
    Returns:
        :return torch.Tensor: The IFFT of the input, with dimension B x 1 or 2 x H x W
    """
    assert data.size(-3) == 2, data.shape
    data = ctoend(data)
    data = ifftshift(data, dim=(-3, -2))
    if not real:
        data = torch.ifft(data, 2, normalized=True)
        data = cto1(data)
    else:
        data = torch.irfft(data, 2, normalized=True, onesided=False).unsqueeze(-3)
    data = fftshift(data, dim=(-2, -1))
    return data


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    :param x: pt.Tenser to be shifted
    :param dim: dimensions to shift along
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def cto1(x):
    """
    Args:
        :param x (tensor) with three last dimensions H x W x (1 or 2)
    Returns:
        :return tensor with three last dimensions (1 or 2) x H x W
    """
    return x.transpose(-3, -1).transpose(-1, -2)


def ctoend(x):
    """
    Args:
        :param x (tensor) with three last dimensions (1 or 2) x H x W
    Returns:
        :return tensor with three last dimensions H x W x (1 or 2)
    """
    return x.transpose(-3, -1).transpose(-3, -2)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    :param x: pt.Tenser to be shifted
    :param dim: dimensions to shift along
    :return:
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    :param x: pt.Tenser to be shifted
    :param dim: dimensions to shift along
    :return:
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def AddComplexZeros(x):
    """ Add complex channel to images filled with zeros to make things compatible with fft2/ifft2"""
    x = x.repeat(1, 2, 1, 1)
    x[:, 1] = 0
    return x


def sample(image, samples_x, samples_y):
    """
    From https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
    """

    # input image is: N x C x H x W
    N, C, H, W = image.shape
    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)
    samples = torch.cat([samples_x, samples_y], 3)
    # samples[:, :, :, 0] = (samples[:, :, :, 0] / (W - 1))  # normalize to between  0 and 1
    # samples[:, :, :, 1] = (samples[:, :, :, 1] / (H - 1))  # normalize to between  0 and 1
    # samples = samples * 2 - 1  # normalize to between -1 and 1

    return torch.nn.functional.grid_sample(image, samples, align_corners=False)


class RadialSpoke(object):
    def __init__(self, degree):
        self.degree = degree

    def get_locations(self, locations_per_shot):
        loc_2d = torch.zeros((2, locations_per_shot))
        loc_2d += torch.linspace(-1, 1, locations_per_shot)
        loc_2d[0] *= np.sin(self.degree / 180 * np.pi)
        loc_2d[1] *= np.cos(self.degree / 180 * np.pi)
        return loc_2d


trajectory_set = lambda locations_per_shot: torch.stack(
    [RadialSpoke(p).get_locations(locations_per_shot) for p in np.arange(0, 360, 1)])


def get_random_ktraj(n_shots=30, locations_per_shot=200):
    M = n_shots
    L = locations_per_shot
    ktraj_all = trajectory_set(L)
    r = torch.randint(len(ktraj_all), size=(1 * M,))
    return ktraj_all[r].reshape(1, M, 2, L).transpose(1, 2).reshape(1, 2, M * L)


class Inverse(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def inverse(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(1, self.z_dim).repeat(self.batch_size, 1))

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()

        self.G.eval()
        print(self.total_step)
        for step in range(start, self.total_step):

            # ================== Train D ================== #

            try:
                real_images, _ = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, _ = next(data_iter)

            # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            real_images = tensor2var(real_images)
            d_out_real, dr1, dr2 = self.D(real_images)
            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            z_dim = self.z_dim
            bs = self.batch_size

            # Create random noise
            class Z(nn.Module):
                def __init__(self):
                    super(Z, self).__init__()

                    self.z = nn.Parameter(tensor2var(torch.randn(1, z_dim).repeat(bs, 1)))

                def forward(self):
                    return self.z

            z = Z()

            #optimizer = torch.optim.SGD(z.parameters(), lr=0.1, momentum=0.9)
            ktraj = get_random_ktraj().cuda()
            ktraj = ktraj.repeat(real_images.shape[0], 1, 1)
            F = lambda x: sample(fft2(AddComplexZeros((x + 0.5) / 2)), ktraj[:, 0], ktraj[:, 1])
            y = F(real_images)

            import matplotlib.pyplot as plt
            plt.imsave('real.png', real_images[4, 0].detach().cpu().numpy(), cmap='gray')
            plt.imsave('real1.png', real_images[5, 0].detach().cpu().numpy(), cmap='gray')
            plt.show()

            GD = False
            if GD:
                optimizer = torch.optim.Adam(z.parameters(), lr=0.05)
                for i in range(3000):
                    fake_images, _, _ = self.G(z())
                    loss = (((y - F(fake_images))**2)).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if i % 100 == 0 or (i < 100 and i %10 == 0):
                        plt.imsave('inverse' + str(i) + '_.png', fake_images[4, 0].detach().cpu().numpy(), cmap='gray')
                        plt.imsave('inverse' + str(i) + '_1.png', fake_images[5, 0].detach().cpu().numpy(), cmap='gray')
                        plt.show()
                        print(i, loss.item())
            else:
                w_opt = torch.optim.SGD(lr=10e-4)
                z_opt = torch.optim.SGD(lr=10e-4)
                sigma = 10e-3
                rho = 1
                fake_images, _, _ = self.G(z())
                w_Gz = w - fake_images
                lmbda = torch.ones(fake_images.shape) * 0.1

                for i in range(0, 3000):
                    Lw = (((y - w)**2)).mean()
                    lagrangian = Lw + (lmbda * w_Gz.flatten()).mean() + rho/2 * ((w_Gz)**2).mean()
                    w_opt.zero_grad()
                    lagrangian.backward()
                    w_opt.step()
                    w_opt.zero_grad()
                    lagrangian.backward()
                    w_opt.step()
                    fake_images, _, _ = self.G(z())
                    w_Gz = w - fake_images

                    sigma = torch.mininum(sigma, sigma / torch.sqrt(((w_Gz)**2).mean())*(i+1)*torch.log(torch.log(i+1)))
                    lmbda = lmbda + sigma * w_Gz
                    print(Lw)
            break
            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                      " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step, d_loss_real.item(),
                             self.G.attn1.gamma.mean().item(), self.G.attn2.gamma.mean().item()))

            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_images, _, _ = self.G(fixed_z)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))

            if (step + 1) % model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

    def build_model(self):

        self.G = Generator(self.batch_size, self.imsize, self.z_dim, self.g_conv_dim).cuda()
        self.D = Discriminator(self.batch_size, self.imsize, self.d_conv_dim).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr,
                                            [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr,
                                            [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        # print(self.G)
        # print(self.D)

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
