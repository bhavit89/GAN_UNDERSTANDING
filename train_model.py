
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dcgan_model import Discriminator , Generator ,initialize_weights



print("IMPORTS WORKING")


# HPYERPARAMETERS 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0002
BATCH_SIZE = 128
IMG_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
FEATURES_DISC = 64
FEATURE_GEN = 64
NUM_EPOCHS = 100
DIRECTORY = "/home/bhavit/Desktop/GAN/Cars Dataset"



transform = transforms.Compose([
    transforms.Resize((IMG_SIZE ,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in  range (CHANNELS_IMG)] ,[0.5 for _ in range(CHANNELS_IMG)]
    ),
])


# Dataset = dataset.ImageFolder(root=DIRECTORY , transform=transform)
Dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms, download=True
)
dataloader = DataLoader(Dataset,batch_size=BATCH_SIZE,shuffle=True)


gen = Generator(NOISE_DIM ,CHANNELS_IMG).to(device)
disc = Discriminator(CHANNELS_IMG).to(device)
initialize_weights(gen)
initialize_weights(disc)

gen_optimizer = optim.Adam(gen.parameters(),lr = LEARNING_RATE ,betas =(0.5,0.999))
disc_optimizer = optim.Adam(disc.parameters(),lr = LEARNING_RATE ,betas =(0.5,0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32,NOISE_DIM,1,1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        disc_optimizer.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        gen_optimizer.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1