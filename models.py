import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_channels):
        super().__init__()

        self.dicriminator = nn.Sequential(
            nn.Conv2d(
                image_channels, 128, kernel_size=4, stride=2, padding=2, bias=False
            ),
            nn.LeakyReLU(0.2),
            self._block(128, 256, 4, 2, 1),
            self._block(256, 512, 4, 2, 1),
            self._block(512, 1024, 4, 2, 1),
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        output = self.dicriminator(x)
        return output.view(output.shape[0], -1)


class Generator(nn.Module):
    def __init__(self, channel_noise, image_channels):
        super().__init__()

        self.fc = nn.Linear(channel_noise, 1024 * 4 * 4)

        self.generator = nn.Sequential(
            # self._block(channel_noise, 1024, 4, 1, 0),
            self._block(1024, 512, 4, 2, 1),
            self._block(512, 256, 4, 2, 1),
            self._block(256, 128, 4, 2, 1),
            nn.ConvTranspose2d(
                128,
                image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.fc(x.view(x.size(0), -1))
        x = x.view(x.size(0), 1024, 4, 4)
        output = self.generator(x)
        return output


def weights_init_normal(model):
    classname = model.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("ConvTranspose") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0.0)


def add_spectral_normalization(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(module)
    return model
