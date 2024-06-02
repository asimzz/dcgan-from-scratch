import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from models import (
    Generator,
    Discriminator,
    add_spectral_normalization,
    weights_init_normal,
)
from utils import load_dataset, get_device, save_batch_images


lr = 2e-4
batch_size = 128
image_dim = 64
channels_dim = 3
noise_dim = 100
betas = (0.5, 0.999)
num_epochs = 10

transform = transforms.Compose(
    [
        transforms.Resize((image_dim, image_dim)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

data_loader = load_dataset(transform=transform, batch_size=batch_size)


generator_model = Generator(channel_noise=noise_dim, image_channels=channels_dim)
discriminator_model = Discriminator(image_channels=channels_dim)
adversarial_loss = nn.BCELoss()

device, cuda = get_device()

if cuda:
    generator_model.cuda()
    discriminator_model.cuda()
    adversarial_loss.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Apply spectral normalization
generator_model = add_spectral_normalization(generator_model)
discriminator_model = add_spectral_normalization(discriminator_model)

# Initialize weights
generator_model.apply(weights_init_normal)
discriminator_model.apply(weights_init_normal)

# Define the optimizers
generator_optimizer = optim.Adam(generator_model.parameters(), lr=lr, betas=betas)
discriminator_optimizer = optim.Adam(
    discriminator_model.parameters(), lr=lr, betas=betas
)

generator_model.train()
discriminator_model.train()


generator_losses = []
discriminator_losses = []
for epoch in range(num_epochs):
    for batch_idx, (images, _) in enumerate(data_loader):

        valid = Tensor(images.size(0), 1).fill_(1.0)
        fake = Tensor(images.size(0), 1).fill_(0.0)

        real_images = images.type(Tensor)

        # Train Generator

        noise_input = Tensor(torch.randn(images.size(0), noise_dim, 1, 1).to(device))

        generated_image = generator_model(noise_input)

        generator_loss = adversarial_loss(discriminator_model(generated_image), valid)

        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        # Train Discriminator

        real_loss = adversarial_loss(discriminator_model(real_images), valid)
        fake_loss = adversarial_loss(
            discriminator_model(generated_image.detach()), fake
        )

        discriminator_loss = (real_loss + fake_loss) / 2

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        discriminator_losses.append(discriminator_loss)
        generator_losses.append(generator_loss)

        print(
            f"[Epoch {epoch}/{num_epochs}] [Batch {batch_idx}/{len(data_loader)}] [D loss: {discriminator_loss.item()}] [G loss: {generator_loss.item()}]"
        )

        with torch.no_grad():
            batches_done = epoch * len(data_loader) + batch_idx
            save_batch_images(
                batches_done=batches_done,
                generated_image=generated_image,
            )
