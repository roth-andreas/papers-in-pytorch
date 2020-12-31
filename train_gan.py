from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch
import models.gan as gan
import matplotlib.pyplot as plt
import torchvision.utils as utils
import math
from tqdm.auto import tqdm


def mnist_dataloader(batch_size=64):
    """
    Creates a dataloader for the MNIST dataset
    Args:
        batch_size: Batch size for the dataloader

    Returns: A dataloader for the MNIST dataset with the batch_size

    """
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
    """
    Generates random noise of shape (batch_size, dim)
    Args:
        batch_size: Number of noise vectors to create
        dim: Dimension of each noise vector

    Returns: Noise Matrix of shape (batch_size, dim)

    """
    return torch.randn((batch_size, dim))


def take_disc_step(discriminator, generator, inputs, noise, criterion):
    """
    Takes a step for the discriminator and returns the loss
    Args:
        discriminator: Discriminator model
        generator: Generator model
        inputs: Real inputs for the Discriminator
        noise: Noise Inputs for the Generator
        criterion: Function for calculating the discriminator loss, that takes in outputs and targets

    Returns: Single loss value tensor for further use in backpropagation

    """
    fake_inputs = generator(noise).detach()
    fake_outputs = discriminator(fake_inputs)
    fake_targets = torch.zeros_like(fake_outputs)

    real_outputs = discriminator(inputs.view(len(inputs), -1))
    real_targets = torch.ones_like(real_outputs)
    disc_loss = (criterion(fake_outputs, fake_targets) + criterion(real_outputs, real_targets)) / 2
    return disc_loss


def take_gen_step(discriminator, generator, noise, criterion):
    """
    Takes a step for the generator and returns the loss
    Args:
        discriminator: Discriminator model
        generator: Generator model
        noise: Noise matrix as an input to the Generator
        criterion: Function for calculating the discriminator loss, that takes in outputs and targets

    Returns: Single loss value tensor for further use in backpropagation

    """
    fake_inputs = generator(noise)
    fake_outputs = discriminator(fake_inputs)
    fake_targets = torch.ones_like(fake_outputs)
    gen_loss = criterion(fake_outputs, fake_targets)
    return gen_loss


def show_results(generator, noise, n_results=16, size=(1, 28, 28)):
    """
    Shows some generated resutls in a seperate window in a non blocking way
    Args:
        generator: Generator model
        noise: Noise matrix for input to the Generator
        n_results: Number of results to display
        size: Size of the resulting images in the form (channels, height, width)

    Returns:

    """
    with torch.no_grad():
        outputs = generator(noise)[:n_results]
    images = utils.make_grid(outputs.view(n_results, *size), nrow=int(math.sqrt(n_results)))
    plt.imshow(images.permute(1, 2, 0).squeeze().cpu())
    plt.show(block=False)
    plt.pause(.1)


def train():
    """
    Trains the gan model with some parameters defined below
    Returns:

    """
    epochs = 100
    lr = 1e-5
    z_dim = 64
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = mnist_dataloader(128)
    generator = gan.Generator(z_dim=z_dim).to(device)
    discriminator = gan.Discriminator().to(device)
    gen_optim = torch.optim.Adam(generator.parameters(), lr=lr)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    n_visual = 25
    fixed_noise = generate_noise(n_visual, z_dim).to(device)
    step = 0
    for epoch in range(epochs):
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            noise = generate_noise(len(inputs), z_dim).to(device)
            disc_loss = take_disc_step(discriminator, generator, inputs, noise, criterion)
            disc_optim.zero_grad()
            disc_loss.backward(retain_graph=True)
            disc_optim.step()

            noise = generate_noise(len(inputs), z_dim).to(device)
            gen_loss = take_gen_step(discriminator, generator, noise, criterion)
            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()
            step += 1
        show_results(generator, fixed_noise, n_visual)
        print(f"Step {step}: Discriminator Loss: {disc_loss:.4f}, Generator Loss: {gen_loss:.4f}")


if __name__ == '__main__':
    train()
