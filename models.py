import torch.nn as nn
import torch
from torchsummary import summary


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, ksize=4, stride=2, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv1d(in_size, out_size, kernel_size=ksize, stride=stride, bias=False, padding_mode='replicate')]
        if normalize:
            layers.append(nn.InstanceNorm1d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, ksize=4, stride=2, output_padding=0, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose1d(in_size, out_size, kernel_size=ksize, stride=stride, output_padding=output_padding, bias=False),
            nn.InstanceNorm1d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 128, normalize=False)
        self.down2 = UNetDown(128, 256)
        self.down3 = UNetDown(256, 512, dropout=0.5)
        self.down4 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUp(512, 512, output_padding=1, dropout=0.5)
        self.up2 = UNetUp(1024, 256, output_padding=0)
        self.up3 = UNetUp(512, 128, output_padding=0)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConstantPad1d((1, 1), 0),
            nn.Conv1d(256, out_channels, 4, padding=2, padding_mode='replicate'),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)

        return self.final(u3)


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, ksize=6, stride=3, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv1d(in_filters, out_filters, ksize, stride=stride, padding_mode='replicate')]
            if normalization:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 128, normalization=False),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv1d(512, 1, 4, bias=False, padding_mode='replicate')
        )

    def forward(self, signal_A, signal_B):
        # Concatenate image and condition image by channels to produce input
        signal_input = torch.cat((signal_A, signal_B), 1)
        return self.model(signal_input)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#generator = GeneratorUNet().to(device)
#summary(generator, input_size=(1, 375))
discriminator = Discriminator().to(device)
summary(discriminator, input_size=[(1, 375), (1, 375)])
