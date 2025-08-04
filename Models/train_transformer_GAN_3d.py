# Transformer-GAN for Time Series Generation
# Author: Omar Hawas
# Dataset: AirQualityUCI (preprocessed_3D)
# Requires: torch, numpy, pandas

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Load preprocessed data
data = pd.read_csv("data\\preprocessed_3D.csv").values
data = data.reshape((-1, 24, 13))  # (samples, time_steps, features)

# DataLoader
batch_size = 64
tensor_data = torch.tensor(data, dtype=torch.float32)
loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
seq_len = 24
features = 13
latent_dim = 32
model_dim = 64
num_heads = 4
epochs = 1000

# === Transformer-Based Generator ===
class TransformerGenerator(nn.Module):
    def __init__(self, latent_dim, seq_len, features):
        super().__init__()
        self.latent_to_embedding = nn.Linear(latent_dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, model_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True),
            num_layers=2
        )
        self.output_layer = nn.Linear(model_dim, features)

    def forward(self, z):
        x = self.latent_to_embedding(z) + self.pos_embedding
        x = self.transformer(x)
        return self.output_layer(x)

# === Discriminator ===
class Discriminator(nn.Module):
    def __init__(self, seq_len, features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * features, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# Instantiate models
generator = TransformerGenerator(latent_dim, seq_len, features).to(device)
discriminator = Discriminator(seq_len, features).to(device)

# Optimizers and loss
opt_gen = optim.Adam(generator.parameters(), lr=1e-4)
opt_disc = optim.Adam(discriminator.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for real_batch, in loader:
        real_batch = real_batch.to(device)

        # === Train Discriminator ===
        z = torch.randn(real_batch.size(0), seq_len, latent_dim).to(device)
        fake_batch = generator(z).detach()
        real_labels = torch.ones(real_batch.size(0), 1).to(device)
        fake_labels = torch.zeros(real_batch.size(0), 1).to(device)

        disc_real = discriminator(real_batch)
        disc_fake = discriminator(fake_batch)
        loss_disc = criterion(disc_real, real_labels) + criterion(disc_fake, fake_labels)

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # === Train Generator ===
        z = torch.randn(real_batch.size(0), seq_len, latent_dim).to(device)
        gen_batch = generator(z)
        disc_pred = discriminator(gen_batch)
        loss_gen = criterion(disc_pred, real_labels)

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_disc.item():.4f} | G Loss: {loss_gen.item():.4f}")

# === Save Generator Model ===
torch.save(generator.state_dict(), "transformer_gan_generator.pth")
print("Transformer-GAN Generator saved to transformer_gan_generator.pth")

# === Generate Synthetic Data and Save to CSV ===
with torch.no_grad():
    z = torch.randn(803, seq_len, latent_dim).to(device)
    synthetic_data = generator(z).cpu().numpy()

synthetic_flat = synthetic_data.reshape((synthetic_data.shape[0], -1))  # (803, 312)
df_synthetic = pd.DataFrame(synthetic_flat)
df_synthetic.to_csv("synthetic_transformer_3D.csv", index=False)
print("✅ Synthetic data saved to synthetic_transformer_3D.csv")
np.save("synthetic_3D.npy", synthetic_data)
np.save("synthetic_3D.npy", synthetic_data)