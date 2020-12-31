import time

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch
from models import dcgan, sagan
import matplotlib.pyplot as plt
import torchvision.utils as utils
import math
from tqdm.auto import tqdm
import numpy as np
import copy

from train import train_helper
from train.train import save_model


def mnist_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataloader = DataLoader(
        MNIST('.',
              download=True,
              transform=transform),
        batch_size=batch_size,
        shuffle=True)
    return dataloader


def generate_noise(batch_size, dim, device):
    return torch.rand((batch_size, dim), device=device)


def wasserstein_gen_step(discriminator, generator, noise, criterion, reals=None):
    fake_inputs = generate_from_noise(generator, noise, reals)
    fake_outputs = discriminator(fake_inputs)

    gen_loss = - torch.mean(fake_outputs)
    return gen_loss


def generate_from_noise(generator, noise, reals=None):
    fake_inputs = generator(noise)
    if reals is not None:
        means = torch.mean(reals[:len(noise)], dim=(1, 2, 3))
        fake_inputs = torch.add(fake_inputs, means.view(-1, 1, 1, 1))
    return fake_inputs


def get_gradient(discriminator, real, fake, epsilon):
    mixed_images = real * epsilon + fake.view(*real.shape) * (1 - epsilon)
    mixed_scores = discriminator(mixed_images)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    return gradient


def wasserstein_disc_step(discriminator, generator, inputs, noise, criterion):
    fake_inputs = generator(noise).detach()
    fake_outputs = discriminator(fake_inputs)

    real_outputs = discriminator(inputs)
    disc_loss = -(torch.mean(real_outputs) - torch.mean(fake_outputs))
    return disc_loss


def wasserstein_loss(fake_outputs, real_outputs=None):
    loss = - torch.mean(fake_outputs)
    if real_outputs is not None:
        loss += torch.mean(real_outputs)
    return loss


def show_results(generator, inputs, n_results=16, size=(1, 28, 28), ax=None, generate=True, reals=None):
    if generate:
        with torch.no_grad():
            inputs = generate_from_noise(generator, inputs[:n_results], reals=reals)[:n_results]

    images = utils.make_grid(inputs.view(n_results, *size), nrow=int(math.sqrt(n_results))).permute(1, 2,
                                                                                                    0).squeeze().cpu()
    if ax is not None:
        ax.cla()
        ax.set_axis_off()
        ax.imshow(images)
        plt.pause(.1)
    fig, axs = plt.subplots(figsize=(10, 10))
    axs.set_axis_off()
    axs.imshow(images, cmap='gray')
    return fig


def show_losses(disc_losses, gen_losses, ax):
    ax.cla()
    ax.plot(disc_losses, 'b', label='Critic Loss')
    ax.plot(gen_losses, 'r', label='Generator Loss')
    ax.legend()
    plt.pause(.01)


from load import load_data
from collections import deque


def train():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    epochs = 100
    disc_lr = 4e-4
    gen_lr = 1e-4
    beta_1 = 0.0
    beta_2 = 0.9
    z_dim = 64
    disc_steps = 1
    dims = (64, 64)
    batch_size = 100
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    train_folders = [
        '200nm_9May14_cropped (complete)',
         #'31Aug2016-100nm (first 1000)',
         #'23Aug2016 100nm (first 1000)',
         #'100nm_27Sep13_exp2 (first 2000)',
         #'200nm_10Apr13 (complete)',
         #'200nm_11Apr13_1 (from 501)',
         #'27July2016-200nm (first 300)',
         #'100nm_27Sep13_exp3 (complete)'
    ]
    _, dataloader = load_data.make_data(
        [f'./pamono-data/{folder}/preprocessed window 100' for folder in train_folders],
        size=dims,
        padding=0,
        window_size=1,
        batch_size=batch_size,
        use_all=True,
        train=False,
        no_particles=True)
    generator = sagan.Generator(z_dim, filter_dim=32, c_channels=1, up_layers=4, spectral_norm=True).to(device)
    discriminator = sagan.Discriminator(in_channels=1, h_channels=32, down_layers=3, spectral_norm=True).to(device)
    gen_optim = torch.optim.Adam(generator.parameters(), lr=gen_lr, betas=(beta_1, beta_2))
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(beta_1, beta_2))
    criterion = wasserstein_loss
    n_visual = 25
    fixed_noise = generate_noise(n_visual, z_dim, device)
    gen_losses = deque([0]*25, 25)
    disc_losses = deque([0]*25, 25)
    name = 'SAGAN_NO_SA_BIG_TransposeConv_9May14'
    gen_name = f'{name}_GEN'
    disc_name = f'{name}_DISC'
    step = 0
    if step > 0:
        train_helper.load_model(gen_name, step - 1, generator, gen_optim)
        train_helper.load_model(disc_name, step - 1, discriminator, disc_optim)
        starting_time = train_helper.load_writer_name(name)
    else:
        starting_time = time.time()

    writer = SummaryWriter(f"runs/{name}-{starting_time}")
    while True:
        batch_disc_loss = []
        batch_gen_loss = []
        for inputs, _, _ in dataloader:
            if step == 0:
                fig = show_results(generator, inputs[:n_visual], n_visual, size=(1, *dims), generate=False)
                writer.add_figure('Actual images',
                                  fig,
                                  global_step=step // disc_steps)
            inputs = inputs.to(device)
            noise = generate_noise(len(inputs), z_dim, device)
            disc_loss = wasserstein_disc_step(discriminator, generator, inputs, noise, criterion)
            batch_disc_loss.append(disc_loss.item())
            disc_optim.zero_grad()
            disc_loss.backward(retain_graph=True)
            disc_optim.step()

            if step % disc_steps == 0:
                noise = generate_noise(len(inputs), z_dim, device)
                gen_loss = wasserstein_gen_step(discriminator, generator, noise, criterion)  # , inputs)
                gen_optim.zero_grad()
                gen_loss.backward()
                gen_optim.step()
                batch_gen_loss.append(gen_loss.item())
            if step % 500 == 0:
                save_model(discriminator, disc_optim, f'trained_models/{disc_name}.pt')
                save_model(generator, gen_optim, f'trained_models/{gen_name}.pt')
                fig = show_results(generator, fixed_noise, n_visual, size=(1, *dims),
                                   ax=ax2)  # , reals=fixed_inputs.to(device))
                writer.add_figure('gan generations',
                                  fig,
                                  global_step=step // disc_steps)
            step += 1
        mean_disc_loss = np.mean(batch_disc_loss)
        mean_gen_loss = np.mean(batch_gen_loss)
        disc_losses.append(mean_disc_loss)
        gen_losses.append(mean_gen_loss)
        show_losses(disc_losses, gen_losses, ax1)
        writer.add_scalar('disc_loss', mean_disc_loss, step // disc_steps)
        writer.add_scalar('gen_loss', mean_gen_loss, step // disc_steps)
        print(f"Step {step}: Discriminator Loss: {mean_disc_loss:.4f}, Generator Loss: {mean_gen_loss:.4f}")
    plt.show()


if __name__ == '__main__':
    train()
