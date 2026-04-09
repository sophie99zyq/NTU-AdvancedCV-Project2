import torch
import torch.nn as nn
import itertools


class Discriminator(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self, in_channels=1):
        super().__init__()
        def block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return layers
        self.model = nn.Sequential(
            *block(in_channels, 64, norm=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1),
        )
    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )
    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """ResNet generator for image translation."""
    def __init__(self, in_channels=1, out_channels=1, n_blocks=6, ngf=64):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3), nn.Conv2d(in_channels, ngf, 7),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),
        ]
        for i in range(2):
            mult = 2 ** i
            model += [nn.Conv2d(ngf*mult, ngf*mult*2, 3, 2, 1),
                      nn.InstanceNorm2d(ngf*mult*2), nn.ReLU(True)]
        mult = 4
        for _ in range(n_blocks):
            model += [ResBlock(ngf * mult)]
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [nn.ConvTranspose2d(ngf*mult, ngf*mult//2, 3, 2, 1, output_padding=1),
                      nn.InstanceNorm2d(ngf*mult//2), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)


def train_cycada(source_loader, target_loader, classifier, num_classes,
                 in_channels=1, n_epochs=50, lr=2e-4, lambda_cyc=10.0,
                 lambda_task=1.0, device='cuda'):
    """Train CyCADA: CycleGAN + task loss.

    Args:
        source_loader: DataLoader for source (images, labels)
        target_loader: DataLoader for target (images, labels unused)
        classifier: pretrained source classifier (for task loss)
        num_classes: number of classes
        in_channels: image channels
        n_epochs: epochs
        lr: learning rate
        lambda_cyc: cycle loss weight
        lambda_task: task loss weight
        device: cuda/cpu
    Returns: G_s2t, G_t2s generators
    """
    G_s2t = ResNetGenerator(in_channels, in_channels).to(device)
    G_t2s = ResNetGenerator(in_channels, in_channels).to(device)
    D_s = Discriminator(in_channels).to(device)
    D_t = Discriminator(in_channels).to(device)
    classifier = classifier.to(device).eval()

    opt_G = torch.optim.Adam(itertools.chain(G_s2t.parameters(), G_t2s.parameters()),
                              lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(D_s.parameters(), D_t.parameters()),
                              lr=lr, betas=(0.5, 0.999))

    criterion_gan = nn.MSELoss()
    criterion_cyc = nn.L1Loss()
    criterion_task = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        for (src_imgs, src_labels), (tgt_imgs, _) in zip(source_loader, target_loader):
            src_imgs = src_imgs.to(device)
            src_labels = src_labels.to(device)
            tgt_imgs = tgt_imgs.to(device)

            batch_size = min(src_imgs.size(0), tgt_imgs.size(0))
            src_imgs, src_labels = src_imgs[:batch_size], src_labels[:batch_size]
            tgt_imgs = tgt_imgs[:batch_size]

            src_norm = src_imgs * 2 - 1
            tgt_norm = tgt_imgs * 2 - 1

            # Train Generators
            opt_G.zero_grad()
            fake_tgt = G_s2t(src_norm)
            fake_src = G_t2s(tgt_norm)

            loss_gan = criterion_gan(D_t(fake_tgt), torch.ones_like(D_t(fake_tgt))) + \
                       criterion_gan(D_s(fake_src), torch.ones_like(D_s(fake_src)))

            recon_src = G_t2s(fake_tgt)
            recon_tgt = G_s2t(fake_src)
            loss_cyc = criterion_cyc(recon_src, src_norm) + criterion_cyc(recon_tgt, tgt_norm)

            fake_tgt_01 = (fake_tgt + 1) / 2
            loss_task = criterion_task(classifier(fake_tgt_01), src_labels)

            loss_G = loss_gan + lambda_cyc * loss_cyc + lambda_task * loss_task
            loss_G.backward()
            opt_G.step()

            # Train Discriminators
            opt_D.zero_grad()
            loss_D_t = (criterion_gan(D_t(tgt_norm), torch.ones_like(D_t(tgt_norm))) +
                        criterion_gan(D_t(fake_tgt.detach()), torch.zeros_like(D_t(fake_tgt.detach())))) * 0.5
            loss_D_s = (criterion_gan(D_s(src_norm), torch.ones_like(D_s(src_norm))) +
                        criterion_gan(D_s(fake_src.detach()), torch.zeros_like(D_s(fake_src.detach())))) * 0.5
            loss_D = loss_D_t + loss_D_s
            loss_D.backward()
            opt_D.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - G: {loss_G.item():.4f} D: {loss_D.item():.4f}")

    return G_s2t, G_t2s


def translate_with_cycada(G_s2t, source_loader, device='cuda'):
    """Translate source images to target domain using trained generator.
    Returns: list of (translated_images, labels) tuples
    """
    G_s2t.eval()
    translated_data = []
    with torch.no_grad():
        for images, labels in source_loader:
            images = images.to(device)
            images_norm = images * 2 - 1
            fake = G_s2t(images_norm)
            fake_01 = (fake + 1) / 2
            translated_data.append((fake_01.cpu(), labels))
    return translated_data
