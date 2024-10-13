import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MapDataset
import config
from utils import save_checkpoint, load_checkpoint, save_examples
from generator import Generator
from discriminator import Discriminator


def train(discriminator, generator, loader, disc_opt, gen_opt, l1, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            y_fake = generator(x)
            D_real = discriminator(x, y)
            D_fake = discriminator(x, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real + D_fake_loss).mean()

        discriminator.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(disc_opt)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            D_fake = discriminator(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        generator.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(gen_opt)
        g_scaler.update()
                


def main():
    discriminator = Discriminator(in_channels=3).to(config.DEVICE)
    generator = Generator(in_channels=3).to(config.DEVICE)
    
    disc_opt = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    gen_opt = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, generator, gen_opt, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, discriminator, disc_opt, config.LEARNING_RATE)

    train_dataset = MapDataset(root="maps/maps/train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    val_dataset = MapDataset(root="maps/maps/val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train(discriminator, generator, train_loader, disc_opt, gen_opt, L1_LOSS, BCE, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(generator, gen_opt, filename=config.CHECKPOINT_GEN)
            save_checkpoint(discriminator, disc_opt, filename=config.CHECKPOINT_DISC)

        save_examples(generator, val_loader, epoch, folder="examples")
    


if __name__ =="__main__":
    main()