import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# --- Load Data ---
data = np.load("data/preprocessed_3D.npy")  # shape: (samples, timesteps, features)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = torch.tensor(data, dtype=torch.float32).to(device)

# --- Hyperparameters ---
latent_dim = 100
epochs = 1000
batch_size = 64
lr = 0.0002
hidden_dim = 128
timesteps = data.shape[1]
features = data.shape[2]

# --- Model Components ---
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn.squeeze(0))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, features, batch_first=True)

    def forward(self, z):
        hidden = self.fc(z).unsqueeze(1).repeat(1, timesteps, 1)
        out, _ = self.lstm(hidden)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(timesteps * features, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Instantiate models
encoder = Encoder().to(device)
decoder = Decoder().to(device)
discriminator = Discriminator().to(device)

# Optimizers
enc_dec_optim = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
disc_optim = optim.Adam(discriminator.parameters(), lr=lr)

# Loss
adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.MSELoss()

# --- Training Loop ---
print("🚀 Training TAEGAN...")

for epoch in range(epochs):
    idx = np.random.permutation(len(data))
    for i in range(0, len(data), batch_size):
        real_seq = data[idx[i:i+batch_size]]

        # === Train Discriminator ===
        z = encoder(real_seq)
        fake_seq = decoder(z).detach()

        real_labels = torch.ones((real_seq.size(0), 1), device=device)
        fake_labels = torch.zeros((fake_seq.size(0), 1), device=device)

        real_pred = discriminator(real_seq)
        fake_pred = discriminator(fake_seq)

        d_loss = adversarial_loss(real_pred, real_labels) + adversarial_loss(fake_pred, fake_labels)

        disc_optim.zero_grad()
        d_loss.backward()
        disc_optim.step()

        # === Train Generator ===
        z = encoder(real_seq)
        recon_seq = decoder(z)
        fake_pred = discriminator(recon_seq)

        g_loss = adversarial_loss(fake_pred, real_labels) + reconstruction_loss(recon_seq, real_seq)

        enc_dec_optim.zero_grad()
        g_loss.backward()
        enc_dec_optim.step()

    print(f"[Epoch {epoch+1}/{epochs}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

# --- Generate synthetic samples ---
with torch.no_grad():
    noise = torch.randn(1000, latent_dim).to(device)
    synthetic_data = decoder(noise).cpu().numpy()

# Save 3D numpy format
npy_path = "data/taegan_generated.npy"
np.save(npy_path, synthetic_data)
print(f"✅ Synthetic 3D data saved to: {npy_path}")

# Flatten to 2D: (samples, timesteps * features)
synthetic_data_2d = synthetic_data.reshape(synthetic_data.shape[0], -1)

# Load original column names
column_reference = pd.read_csv("data/preprocessed_3D.csv")
columns = list(column_reference.columns)

# Create DataFrame and save as CSV
csv_path = "data/taegan_generated_data.csv"
synthetic_df = pd.DataFrame(synthetic_data_2d, columns=columns)
synthetic_df.to_csv(csv_path, index=False)
print(f"✅ Synthetic 2D CSV saved to: {csv_path}")
