import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# --- 1. Load and preprocess your 2D CSV data ---
df = pd.read_csv("data\preprocessed_2D.csv")
scaler = MinMaxScaler()
data = scaler.fit_transform(df.values)  # Normalize to [0, 1]

tensor_data = torch.tensor(data, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

input_dim = data.shape[1]
latent_dim = 32

# --- 2. Define models ---

class DenseEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x): return self.encoder(x)

class DenseDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, z): return self.decoder(z)

class DenseGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, z): return self.model(z)

class DenseDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.model(x)

# --- 3. Instantiate and move to device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = DenseEncoder(input_dim, latent_dim).to(device)
decoder = DenseDecoder(latent_dim, input_dim).to(device)
generator = DenseGenerator(latent_dim, input_dim).to(device)
discriminator = DenseDiscriminator(input_dim).to(device)

# --- 4. Optimizers and losses ---
criterion = nn.BCELoss()
recon_loss = nn.MSELoss()
opt_auto = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
opt_g = optim.Adam(generator.parameters(), lr=1e-4)
opt_d = optim.Adam(discriminator.parameters(), lr=1e-4)

# --- 5. Training loop ---
epochs = 1000
for epoch in range(epochs):
    for batch in loader:
        real_data = batch[0].to(device)

        # --- Autoencoder Training ---
        opt_auto.zero_grad()
        z = encoder(real_data)
        recon = decoder(z)
        loss_ae = recon_loss(recon, real_data)
        loss_ae.backward()
        opt_auto.step()

        # --- Discriminator Training ---
        opt_d.zero_grad()
        real_labels = torch.ones(real_data.size(0), 1).to(device)
        fake_labels = torch.zeros(real_data.size(0), 1).to(device)

        noise = torch.randn(real_data.size(0), latent_dim).to(device)
        fake_data = generator(noise).detach()

        d_real = discriminator(real_data)
        d_fake = discriminator(fake_data)

        d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
        d_loss.backward()
        opt_d.step()

        # --- Generator Training ---
        opt_g.zero_grad()
        gen_data = generator(noise)
        g_loss = criterion(discriminator(gen_data), real_labels)
        g_loss.backward()
        opt_g.step()

    # --- Logging ---
    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:04d} | AE Loss: {loss_ae:.4f} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

# --- 6. Save generated data ---
with torch.no_grad():
    z = torch.randn(1000, latent_dim).to(device)
    synthetic = generator(z).cpu().numpy()
    synthetic = scaler.inverse_transform(synthetic)
    pd.DataFrame(synthetic, columns=df.columns).to_csv("data\TAEGAN_generated_data_2D.csv", index=False)
