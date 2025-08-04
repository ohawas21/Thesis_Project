import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# --- 1. Load and normalize CSV ---
df = pd.read_csv("data\preprocessed_2D.csv")
scaler = MinMaxScaler()
data = scaler.fit_transform(df.values)

tensor_data = torch.tensor(data, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

input_dim = data.shape[1]
latent_dim = 32

# --- 2. Transformer-based Generator ---
class TransformerGenerator(nn.Module):
    def __init__(self, latent_dim, seq_len):
        super().__init__()
        self.embedding = nn.Linear(latent_dim, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=seq_len, nhead=1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.project = nn.Linear(seq_len, seq_len)

    def forward(self, z):
        x = self.embedding(z).unsqueeze(0)  # [1, batch, seq_len]
        x = self.transformer(x)
        return self.project(x.squeeze(0))  # [batch, seq_len]

# --- 3. Discriminator ---
class TransformerDiscriminator(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.proj = nn.Linear(seq_len, 128)
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.proj(x)
        return self.classifier(x)

# --- 4. Instantiate Models ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = TransformerGenerator(latent_dim, input_dim).to(device)
D = TransformerDiscriminator(input_dim).to(device)

# --- 5. Losses and Optimizers ---
loss_fn = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=1e-4)
opt_D = optim.Adam(D.parameters(), lr=1e-4)

# --- 6. Training Loop ---
epochs = 1000
for epoch in range(epochs):
    for batch in loader:
        real_data = batch[0].to(device)

        # Train Discriminator
        z = torch.randn(real_data.size(0), latent_dim).to(device)
        fake_data = G(z).detach()
        real_labels = torch.ones(real_data.size(0), 1).to(device)
        fake_labels = torch.zeros(real_data.size(0), 1).to(device)

        opt_D.zero_grad()
        d_loss = loss_fn(D(real_data), real_labels) + loss_fn(D(fake_data), fake_labels)
        d_loss.backward()
        opt_D.step()

        # Train Generator
        z = torch.randn(real_data.size(0), latent_dim).to(device)
        gen_data = G(z)
        opt_G.zero_grad()
        g_loss = loss_fn(D(gen_data), real_labels)
        g_loss.backward()
        opt_G.step()

    # Logging
    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:04d} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

# --- 7. Save Synthetic Data ---
with torch.no_grad():
    z = torch.randn(1000, latent_dim).to(device)
    synthetic = G(z).cpu().numpy()
    synthetic = scaler.inverse_transform(synthetic)
    pd.DataFrame(synthetic, columns=df.columns).to_csv("TransformersGAN_generated_data_2D.csv", index=False)
