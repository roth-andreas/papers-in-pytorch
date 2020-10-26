from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch
import models.gan as gan
from models import dcgan
import matplotlib.pyplot as plt
import torchvision.utils as utils
import math
from tqdm.auto import tqdm
import numpy as np


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
    return torch.randn((batch_size, dim), device=device)


def wasserstein_gen_step(discriminator, generator, noise, criterion):
    fake_inputs = generator(noise)
    fake_outputs = discriminator(fake_inputs)

    gen_loss = - torch.mean(fake_outputs)
    return gen_loss

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
    epsilon = torch.rand(len(inputs), 1, 1, 1, device="cuda", requires_grad=True)
    gradient = get_gradient(discriminator, inputs, fake_inputs, epsilon)
    gp = gradient_penalty(gradient)
    gp_lambda = 10
    disc_loss = -(torch.mean(real_outputs) - torch.mean(fake_outputs)) + gp_lambda * gp
    return disc_loss


def wasserstein_loss(fake_outputs, real_outputs=None):
    loss = - torch.mean(fake_outputs)
    if real_outputs is not None:
        loss += torch.mean(real_outputs)
    return loss


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def show_results(generator, noise, n_results=16, size=(1, 28, 28), ax=None):

    with torch.no_grad():
        outputs = generator(noise)[:n_results]
    outputs = (outputs + 1) / 2
    images = utils.make_grid(outputs.view(n_results, *size), nrow=int(math.sqrt(n_results)))
    ax.cla()
    ax.imshow(images.permute(1, 2, 0).squeeze().cpu())
    plt.pause(.1)


def show_losses(disc_losses, gen_losses, ax):
    ax.cla()
    ax.plot(disc_losses, 'b', label='Critic Loss')
    ax.plot(gen_losses, 'r', label='Generator Loss')
    ax.legend()
    plt.pause(.01)


def train():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    epochs = 100
    lr = 2e-4
    beta_1 = 0.5
    beta_2 = 0.999
    z_dim = 64
    disc_steps = 5
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = mnist_dataloader(128)
    generator = dcgan.Generator(z_dim=z_dim).to(device)
    discriminator = dcgan.Discriminator().to(device)
    gen_optim = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))
    criterion = wasserstein_loss
    n_visual = 25
    fixed_noise = generate_noise(n_visual, z_dim, device)
    step = 0
    gen_losses = []
    disc_losses = []
    for epoch in range(epochs):
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            noise = generate_noise(len(inputs), z_dim, device)
            disc_loss = wasserstein_disc_step(discriminator, generator, inputs, noise, criterion)
            disc_losses.append(disc_loss.item())
            disc_optim.zero_grad()
            disc_loss.backward(retain_graph=True)
            disc_optim.step()

            if step % disc_steps == 0:
                noise = generate_noise(len(inputs), z_dim, device)
                gen_loss = wasserstein_gen_step(discriminator, generator, noise, criterion)
                gen_optim.zero_grad()
                gen_loss.backward()
                gen_optim.step()
                show_losses(disc_losses, gen_losses, ax1)
            gen_losses.append(gen_loss.item())
            step += 1
            if step % 100 == 0:
                show_results(generator, fixed_noise, n_visual, ax=ax2)
        print(f"Step {step}: Discriminator Loss: {disc_loss:.4f}, Generator Loss: {gen_loss:.4f}")
    plt.show()


if __name__ == '__main__':
    train()
