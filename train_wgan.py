from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch
import models.gan as gan
import matplotlib.pyplot as plt
import torchvision.utils as utils
import math
from tqdm.auto import tqdm
import numpy as np


def mnist_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataloader = DataLoader(
        MNIST('.',
              download=True,
              transform=transform),
        batch_size=batch_size,
        shuffle=True)
    return dataloader


def generate_noise(batch_size, dim):
    return torch.randn((batch_size, dim))


def wasserstein_gen_step(discriminator, generator, noise, criterion):
    fake_inputs = generator(noise).detach()
    fake_outputs = discriminator(fake_inputs)

    gen_loss = - torch.mean(fake_outputs)
    return gen_loss


def wasserstein_disc_step(discriminator, generator, inputs, noise, criterion):
    fake_inputs = generator(noise).detach()
    fake_outputs = discriminator(fake_inputs)

    real_outputs = discriminator(inputs)

    disc_loss = torch.mean(real_outputs) - torch.mean(fake_outputs)
    return - disc_loss


def wasserstein_loss(fake_outputs, real_outputs=None):
    loss = - torch.mean(fake_outputs)
    if real_outputs is not None:
        loss += torch.mean(real_outputs)
    return loss


def show_results(generator, noise, n_results=16, size=(1, 28, 28), ax=None):

    with torch.no_grad():
        outputs = generator(noise)[:n_results]
    images = utils.make_grid(outputs.view(n_results, *size), nrow=int(math.sqrt(n_results)))
    ax.imshow(images.permute(1, 2, 0).squeeze().cpu())
    plt.pause(.01)


def show_losses(disc_losses, gen_losses, ax):
    ax.plot(disc_losses, 'b', label='Critic Loss')
    ax.plot(gen_losses, 'r', label='Generator Loss')
    plt.pause(.01)


def train():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    epochs = 100
    lr = 1e-5
    z_dim = 64
    disc_steps = 5
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = mnist_dataloader(128)
    generator = gan.Generator(z_dim=z_dim).to(device)
    discriminator = gan.Discriminator().to(device)
    gen_optim = torch.optim.RMSprop(generator.parameters(), lr=lr)
    disc_optim = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
    criterion = wasserstein_loss
    n_visual = 25
    fixed_noise = generate_noise(n_visual, z_dim).to(device)
    step = 0
    gen_losses = []
    disc_losses = []
    for epoch in range(epochs):
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            noise = generate_noise(len(inputs), z_dim).to(device)
            disc_loss = wasserstein_disc_step(discriminator, generator, inputs, noise, criterion)
            disc_losses.append(disc_loss.item())
            disc_optim.zero_grad()
            disc_loss.backward(retain_graph=True)
            disc_optim.step()
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            if step % disc_steps == 0:
                noise = generate_noise(len(inputs), z_dim).to(device)
                gen_loss = wasserstein_gen_step(discriminator, generator, noise, criterion)
                gen_optim.zero_grad()
                gen_loss.backward()
                gen_optim.step()
                for p in generator.parameters():
                    p.data.clamp_(-0.01, 0.01)
                show_losses(disc_losses, gen_losses, ax1)
            gen_losses.append(gen_loss.item())
            step += 1
            if step % 250 == 0:
                show_results(generator, fixed_noise, n_visual, ax=ax2)
        print(f"Step {step}: Discriminator Loss: {disc_loss:.4f}, Generator Loss: {gen_loss:.4f}")
    plt.show()


if __name__ == '__main__':
    train()
