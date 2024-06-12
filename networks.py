import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torch import nn, optim
from itertools import combinations, chain
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import ABC, abstractmethod


def weights_init(m):
    """For initializing weights from normal distribution"""
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class BaseModel(ABC):
    """Abstract base class for model"""

    def __init__(self, gens, discs):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.criterionGAN = nn.MSELoss().to(self.device)

        # optimizers
        gen_params = chain(*[gen.parapetrs() for gen in gens])
        disc_params = chain(*[disc.parapetrs() for disc in discs])
        self.gen_opt = optim.Adam(gen_params, lr=2e-4, betas=(0.5, 0.999))
        self.disc_opt = optim.Adam(disc_params, lr=2e-4, betas=(0.5, 0.999))

        # schedulers
        self.gen_scheduler = optim.lr_scheduler.StepLR(self.gen_opt, 100, gamma=0.1)
        self.dis_scheduler = optim.lr_scheduler.StepLR(self.disc_opt, 100, gamma=0.1)

    @abstractmethod
    def forward(self):
        """Forward pass through model"""
        pass

    @abstractmethod
    def backward_gen(self):
        """Calculate losses, gradients, and update generators weights"""
        pass

    def backward_disc_basic(self, disc, real, fake):
        """Calculate GAN loss for discriminator"""
        pred_real = disc(real)
        loss_real = self.criterionGAN(pred_real, torch.full_like(pred_real, 0.9))
        loss_real.backward()

        pred_fake = disc(fake.detach())
        loss_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        loss_fake.backward()

        loss = (loss_real + loss_fake) * 0.5
        return loss

    @abstractmethod
    def train_epoch(self, loaderA, loaderB):
        """Feed data to the model once"""
        pass

    def train(self, epochs, loaderA, loaderB):
        """Train model for some epochs"""
        for i in range(epochs):
            print(f"Epoch {i + 1}\n-------------------------")
            self.train_epoch(loaderA, loaderB)
            self.gen_scheduler.step()
            self.dis_scheduler.step()
        self.save(epochs)

    def set_requires_grad(self, nets, requires_grad):
        """Freeze/unfreeze part of the model"""
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    @abstractmethod
    def sample(self):
        """Visualize some samples"""
        pass

    @abstractmethod
    def save(self, iter):
        """Save model to the file"""
        pass

    @abstractmethod
    def load(self, path):
        """Load model from the file"""
        pass


class ResidualBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, num_feat):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
                   nn.BatchNorm2d(num_feat),
                   nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
                   nn.BatchNorm2d(num_feat)]

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        x = x + out
        return x


class Generator(nn.Module):
    """Resnet generator"""

    def __init__(self, in_channels, num_feat, num_res):
        super().__init__()

        layers = []
        layers += [
                nn.Conv2d(in_channels, num_feat, kernel_size=7, stride=1, padding=3, padding_mode='reflect',
                          bias=False),
                nn.BatchNorm2d(num_feat),
                nn.ReLU(inplace=True)
        ]

        curr_dim = num_feat
        for _ in range(2):
            layers += [nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(curr_dim * 2),
                       nn.ReLU(inplace=True)]
            curr_dim = curr_dim * 2

        for _ in range(num_res):
            layers += [ResidualBlock(curr_dim)]

        for _ in range(2):
            layers += [nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1,
                                          bias=False),
                       nn.BatchNorm2d(curr_dim // 2),
                       nn.ReLU(inplace=True)]
            curr_dim = curr_dim // 2

        layers += [
                nn.Conv2d(curr_dim, in_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect',
                          bias=False),
                nn.Tanh()
        ]

        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN"""

    def __init__(self, in_channels, num_feat=64, num_repeat=4):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels, num_feat, kernel_size=4, stride=2, padding=1),
                   nn.LeakyReLU(0.2, inplace=True)]

        curr_dim = num_feat
        for _ in range(1, num_repeat):
            layers += [nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1),
                       nn.BatchNorm2d(curr_dim * 2),
                       nn.LeakyReLU(0.2, inplace=True)]
            curr_dim = curr_dim * 2

        layers += [nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=1, padding=1),
                   nn.BatchNorm2d(curr_dim),
                   nn.LeakyReLU(0.2, inplace=True)]

        layers += [nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1)]

        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)


class SiameseNet(nn.Module):
    """Conv-net for exploring features in a latent space"""

    def __init__(self, image_size, in_channels, num_feat=64, num_repeat=5, delta=10):
        super().__init__()

        self.delta = delta

        layers = []
        layers += [nn.Conv2d(in_channels, num_feat, kernel_size=4, stride=2, padding=1),
                   nn.LeakyReLU(0.02, inplace=True)]

        curr_dim = num_feat
        for _ in range(1, num_repeat):
            layers += [nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1),
                       nn.LeakyReLU(0.02, inplace=True)]
            curr_dim = curr_dim * 2

        in_feat = image_size // 2 ** num_repeat

        self.main = nn.Sequential(*layers)
        self.linear = nn.Linear(curr_dim * in_feat ** 2, 1024)
        self.apply(weights_init)

    def forward(self, x1, x2):
        latent1 = self.main(x1)
        latent2 = self.main(x2)
        latent1 = self.linear(latent1.flatten(1))
        latent2 = self.linear(latent2.flatten(1))
        return latent1, latent2

    def calc_loss(self, x1, x2):
        """Calculate siamese loss"""
        pairs = np.asarray(list(combinations(list(range(x1.size(0))), 2)))
        latent1, latent2 = self.forward(x1, x2)
        v1 = latent1[pairs[:, 0]] - latent1[pairs[:, 1]]
        v2 = latent2[pairs[:, 0]] - latent2[pairs[:, 1]]
        distance = torch.mean(F.cosine_similarity(v1, v2, dim=-1))
        margin = self.margin_loss(v1)
        return margin - distance

    def margin_loss(self, v1):
        """Calculate margin loss"""
        return torch.mean(F.relu(self.delta - torch.norm(v1, dim=1)))


class DINOWrapper(nn.Module):
    """Wrapper for DINOv2 as a siamese net"""

    def __init__(self, backbone):
        super().__init__()
        self.main = backbone

    def forward(self, x1, x2):
        latent1 = self.main(x1).last_hidden_state[:, 1:, :]
        latent2 = self.main(x2).last_hidden_state[:, 1:, :]
        return latent1, latent2

    def calc_loss(self, x1, x2):
        """Calculate siamese loss"""
        pairs = np.asarray(list(combinations(list(range(x1.size(0))), 2)))
        latent1, latent2 = self.forward(x1, x2)
        v1 = latent1[pairs[:, 0]] - latent1[pairs[:, 1]]
        v2 = latent2[pairs[:, 0]] - latent2[pairs[:, 1]]
        distance = torch.mean(F.cosine_similarity(v1, v2, dim=-1))
        return -distance


class TravelGan(BaseModel):
    """TraVeLGAN"""
    def __init__(self, gen, disc, siamese):
        super().__init__([gen, siamese], [disc])

        self.gen = gen.to(self.device)
        self.disc = disc.to(self.device)
        self.siamese = siamese.to(self.device)

    def forward(self):
        self.fake_y = self.gen(self.real_x)

    def backward_gen(self):
        pred = self.disc(self.fake_y)
        gen_loss = self.criterionGAN(pred, torch.ones_like(pred))
        gen_loss.backward()

        return gen_loss

    def backward_siamese(self):
        """Calculate siamese loss, gradients and update siamese weights"""
        siamese_loss = self.siamese.calc_loss(self.real_x, self.fake_y)
        siamese_loss.backward()

        return siamese_loss

    def train_epoch(self, loaderA, loaderB):
        G_losses = []
        D_losses = []
        S_losses = []

        for i, (x_a, x_b) in tqdm(enumerate(zip(loaderA, loaderB))):

            if isinstance(x_a, (tuple, list)):
                x_a = x_a[0]
            if isinstance(x_b, (tuple, list)):
                x_b = x_b[0]

            self.real_x = x_a.to(self.device)
            self.real_y = x_b.to(self.device)

            # forward
            self.forward()

            # optimize gens
            self.set_requires_grad([self.disc], False)
            self.gen_opt.zero_grad()
            gen_loss = self.backward_gen()
            siamese_loss = self.backward_siamese()
            self.gen_opt.step()

            # optimize discs
            self.set_requires_grad([self.disc], True)
            self.disc_opt.zero_grad()
            disc_loss = self.backward_disc_basic(self.disc, self.real_x, self.fake_y)
            self.disc_opt.step()

            G_losses.append(gen_loss.item())
            D_losses.append(disc_loss.item())
            S_losses.append(siamese_loss.item())

        print(
            f"G_loss: {sum(G_losses) / len(G_losses)}, "
            f"D_loss: {sum(D_losses) / len(D_losses)}, "
            f"S_loss: {sum(S_losses) / len(S_losses)}"
        )
        self.sample()

    def sample(self):
        _, ax = plt.subplots(4, 2, figsize=(10, 15))
        for i in range(4):
            ax[i, 0].imshow(v2.functional.to_pil_image(self.real_x[i] * 0.5 + 0.5))
            ax[i, 1].imshow(v2.functional.to_pil_image(self.fake_y[i] * 0.5 + 0.5))
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
        plt.show()
        plt.close()

    def save(self, iter):
        torch.save({'gen': self.gen.state_dict(),
                    'disc': self.disc.state_dict(),
                    'siamese': self.siamese.state_dict(),
                    'gen_opt': self.gen_opt.state_dict(),
                    'disc_opt': self.disc_opt.state_dict(),
                    'gen_scheduler': self.gen_scheduler.state_dict(),
                    'dis_scheduler': self.dis_scheduler.state_dict()
                    }, f'checkpoints/travelgan_{iter}.pt')

    def load(self, path):
        checkpoint = torch.load(path)
        self.gen.load_state_dict(checkpoint['gen'])
        self.disc.load_state_dict(checkpoint['disc'])
        self.siamese.load_state_dict(checkpoint['siamese'])
        self.disc_opt.load_state_dict(checkpoint['disc_opt'])
        self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        self.gen_scheduler.load_state_dict(checkpoint['gen_scheduler'])
        self.dis_scheduler.load_state_dict(checkpoint['dis_scheduler'])


class CycleGAN(BaseModel):
    """CycleGAN"""

    def __init__(self, genG, genF, discX, discY):
        super().__init__([genG, genF], [discX, discY])

        # networks
        self.genG = genG.to(self.device)
        self.genF = genF.to(self.device)
        self.discX = discX.to(self.device)
        self.discY = discY.to(self.device)

        # losses
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()

        # lambdas
        self.lambda_cyc = 10.0
        self.lambda_idt = 0.5

    def forward(self):
        self.fake_y = self.genG(self.real_x)
        self.cyc_x = self.genF(self.fake_y)
        self.fake_x = self.genF(self.real_y)
        self.cyc_y = self.genG(self.fake_x)

    def backward_gen(self):
        if self.lambda_idt > 0:
            idt_y = self.genG(self.real_y)
            loss_idt_G = self.criterionIdt(idt_y, self.real_y) * self.lambda_cyc * self.lambda_idt
            idt_x = self.genF(self.real_x)
            loss_idt_F = self.criterionIdt(idt_x, self.real_x) * self.lambda_cyc * self.lambda_idt
        else:
            loss_idt_G = 0
            loss_idt_F = 0

        discY_fake_y = self.discY(self.fake_y)
        loss_G = self.criterionGAN(discY_fake_y, torch.ones_like(discY_fake_y))

        discX_fake_x = self.discX(self.fake_x)
        loss_F = self.criterionGAN(discX_fake_x, torch.ones_like(discX_fake_x))

        loss_cycle_G = self.criterionCycle(self.cyc_y, self.real_y) * self.lambda_cyc

        loss_cycle_F = self.criterionCycle(self.cyc_x, self.real_x) * self.lambda_cyc

        loss_G_total = loss_G + loss_cycle_G + loss_idt_G
        loss_F_total = loss_F + loss_cycle_F + loss_idt_F

        gens_loss = loss_G_total + loss_F_total
        gens_loss.backward()

        return loss_G_total, loss_F_total

    def backward_discY(self):
        return self.backward_disc_basic(self.discY, self.real_y, self.fake_y)

    def backward_discX(self):
        return self.backward_disc_basic(self.discX, self.real_x, self.fake_x)

    def train_epoch(self, loaderA, loaderB):
        G_losses = []
        F_losses = []
        D_X_losses = []
        D_Y_losses = []

        for i, (x, y) in tqdm(enumerate(zip(loaderA, loaderB))):

            if isinstance(x, (tuple, list)):
                x = x[0]
            if isinstance(y, (tuple, list)):
                y = y[0]

            self.real_x = x.to(self.device)
            self.real_y = y.to(self.device)

            # forward
            self.forward()

            # optimize gens
            self.set_requires_grad([self.discX, self.discY], False)
            self.gen_opt.zero_grad()
            G_loss, F_loss = self.backward_gens()
            self.gen_opt.step()

            # optimize discs
            self.set_requires_grad([self.discX, self.discY], True)
            self.disc_opt.zero_grad()
            D_Y_loss = self.backward_discY()
            D_X_loss = self.backward_discX()
            self.disc_opt.step()

            G_losses.append(G_loss.item())
            F_losses.append(F_loss.item())
            D_X_losses.append(D_X_loss.item())
            D_Y_losses.append(D_Y_loss.item())

        print(
            (f"G_loss: {sum(G_losses) / len(G_losses)}, "
             f"F_loss: {sum(F_losses) / len(F_losses)}, "
             f"D_X_loss: {sum(D_X_losses) / len(D_X_losses)}, "
             f"D_Y_loss: {sum(D_Y_losses) / len(D_Y_losses)}")
        )
        self.sample()

    def sample(self):
        _, ax = plt.subplots(3, 3, figsize=(10, 15))
        for i in range(3):
            ax[i, 0].imshow(v2.functional.to_pil_image(self.real_x[i] * 0.5 + 0.5))
            ax[i, 1].imshow(v2.functional.to_pil_image(self.fake_y[i] * 0.5 + 0.5))
            ax[i, 2].imshow(v2.functional.to_pil_image(self.cyc_x[i] * 0.5 + 0.5))
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
            ax[i, 2].axis("off")
        plt.show()
        plt.close()

    def save(self, iter):
        torch.save({'genG': self.genG.state_dict(),
                    'genF': self.genF.state_dict(),
                    'discX': self.discX.state_dict(),
                    'discY': self.discY.state_dict(),
                    'gen_opt': self.gen_opt.state_dict(),
                    'disc_opt': self.disc_opt.state_dict(),
                    'gen_scheduler': self.gen_scheduler.state_dict(),
                    'dis_scheduler': self.dis_scheduler.state_dict()
                    }, f'checkpoints/cyclegan_{iter}.pt')

    def load(self, path):
        checkpoint = torch.load(path)
        self.genG.load_state_dict(checkpoint['genG'])
        self.genF.load_state_dict(checkpoint['genF'])
        self.discX.load_state_dict(checkpoint['discX'])
        self.discY.load_state_dict(checkpoint['discY'])
        self.disc_opt.load_state_dict(checkpoint['disc_opt'])
        self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        self.gen_scheduler.load_state_dict(checkpoint['gen_scheduler'])
        self.dis_scheduler.load_state_dict(checkpoint['dis_scheduler'])
