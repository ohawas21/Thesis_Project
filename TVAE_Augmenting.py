import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class CycleFeatureExtractor:
    """Extract statistical and temporal features from molding cycles (same as CT-GAN)"""
    
    def __init__(self, critical_params=None):
        if critical_params is None:
            self.critical_params = [
                'Spritzdruck', 'IST-Schliesskraft', 'EE-Position',
                'SC-Position', 'SC-Drehzahl', 'SC-Drehmoment', 'Anlagekraft',
                'WZ-Position', 'Kreuzkopf-Position', 'EE-Geschwindigkeit'
            ]
        else:
            self.critical_params = critical_params
    
    def extract_cycle_features(self, cycle_data):
        """Extract comprehensive statistical features from a single cycle"""
        features = {}
        
        for param in self.critical_params:
            if param in cycle_data.columns:
                values = cycle_data[param].values
                
                # Basic statistics
                features[f'{param}_mean'] = np.mean(values)
                features[f'{param}_std'] = np.std(values)
                features[f'{param}_min'] = np.min(values)
                features[f'{param}_max'] = np.max(values)
                features[f'{param}_median'] = np.median(values)
                features[f'{param}_range'] = np.max(values) - np.min(values)
                
                # Distribution characteristics
                features[f'{param}_skewness'] = stats.skew(values)
                features[f'{param}_kurtosis'] = stats.kurtosis(values)
                features[f'{param}_q25'] = np.percentile(values, 25)
                features[f'{param}_q75'] = np.percentile(values, 75)
                
                # Temporal characteristics
                if len(values) > 1:
                    slope, _ = np.polyfit(range(len(values)), values, 1)
                    features[f'{param}_trend'] = slope
                    features[f'{param}_start'] = values[0]
                    features[f'{param}_end'] = values[-1]
                    features[f'{param}_change'] = values[-1] - values[0]
                    features[f'{param}_cv'] = np.std(values) / np.abs(np.mean(values)) if np.mean(values) != 0 else 0
                    
                    diff_values = np.diff(values)
                    features[f'{param}_mean_derivative'] = np.mean(diff_values)
                    features[f'{param}_std_derivative'] = np.std(diff_values)
        
        features['cycle_length'] = len(cycle_data)
        features['cycle_duration_ratio'] = len(cycle_data) / 3000
        
        return features

class TVAEEncoder(nn.Module):
    """TVAE Encoder - maps data to latent space"""
    
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dims=[128, 64]):
        super(TVAEEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Build encoder network
        layers = []
        prev_dim = input_dim + condition_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent space parameters
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x, condition):
        # Concatenate input and condition
        encoder_input = torch.cat([x, condition], dim=1)
        
        # Pass through encoder
        h = self.encoder(encoder_input)
        
        # Get latent parameters
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        return mu, logvar

class TVAEDecoder(nn.Module):
    """TVAE Decoder - reconstructs data from latent space"""
    
    def __init__(self, latent_dim, condition_dim, output_dim, hidden_dims=[64, 128]):
        super(TVAEDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        
        # Build decoder network
        layers = []
        prev_dim = latent_dim + condition_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z, condition):
        # Concatenate latent vector and condition
        decoder_input = torch.cat([z, condition], dim=1)
        
        # Reconstruct data
        reconstructed = self.decoder(decoder_input)
        
        return reconstructed

class MoldingQualityTVAE(nn.Module):
    """Complete TVAE model for molding quality data"""
    
    def __init__(self, input_dim, condition_dim=2, latent_dim=32):
        super(MoldingQualityTVAE, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Initialize encoder and decoder
        self.encoder = TVAEEncoder(input_dim, condition_dim, latent_dim)
        self.decoder = TVAEDecoder(latent_dim, condition_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, condition):
        # Encode
        mu, logvar = self.encoder(x, condition)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z, condition)
        
        return reconstructed, mu, logvar

class CompleteMoldingTVAE:
    """Complete TVAE implementation for molding quality data"""
    
    def __init__(self, acceptable_folder, non_acceptable_folder):
        self.acceptable_folder = acceptable_folder
        self.non_acceptable_folder = non_acceptable_folder
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = CycleFeatureExtractor()
        
        print(f"Using device: {self.device}")
    
    def load_and_extract_features(self):
        """Load all data and extract statistical features"""
        print("Loading and extracting features for TVAE...")
        
        features_list = []
        labels_list = []
        
        # Process acceptable files
        print(f"Processing acceptable files...")
        try:
            acceptable_files = [f for f in os.listdir(self.acceptable_folder) if f.endswith('.csv')]
            print(f"Found {len(acceptable_files)} acceptable CSV files")
            
            for file_name in acceptable_files:
                file_path = os.path.join(self.acceptable_folder, file_name)
                try:
                    data = pd.read_csv(file_path)
                    cycles = data['Zykluszahl'].unique()
                    
                    for cycle_id in cycles:
                        cycle_data = data[data['Zykluszahl'] == cycle_id]
                        features = self.feature_extractor.extract_cycle_features(cycle_data)
                        features_list.append(features)
                        labels_list.append(1)  # Acceptable = 1
                        
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
        except Exception as e:
            print(f"Error accessing acceptable folder: {e}")
        
        # Process non-acceptable files
        print(f"Processing non-acceptable files...")
        try:
            non_acceptable_files = [f for f in os.listdir(self.non_acceptable_folder) if f.endswith('.csv')]
            print(f"Found {len(non_acceptable_files)} non-acceptable CSV files")
            
            for file_name in non_acceptable_files:
                file_path = os.path.join(self.non_acceptable_folder, file_name)
                try:
                    data = pd.read_csv(file_path)
                    cycles = data['Zykluszahl'].unique()
                    
                    for cycle_id in cycles:
                        cycle_data = data[data['Zykluszahl'] == cycle_id]
                        features = self.feature_extractor.extract_cycle_features(cycle_data)
                        features_list.append(features)
                        labels_list.append(0)  # Non-acceptable = 0
                        
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
        except Exception as e:
            print(f"Error accessing non-acceptable folder: {e}")
        
        # Convert to DataFrame
        if not features_list:
            raise ValueError("No features extracted! Please check your folder paths and CSV files.")
        
        self.features_df = pd.DataFrame(features_list).fillna(0)
        self.labels = np.array(labels_list)
        
        print(f"Extracted features from {len(features_list)} cycles")
        print(f"Feature dimensions: {self.features_df.shape}")
        print(f"Class distribution: Acceptable={np.sum(self.labels)} ({np.sum(self.labels)/len(self.labels)*100:.1f}%), Non-acceptable={len(self.labels) - np.sum(self.labels)} ({(1-np.sum(self.labels)/len(self.labels))*100:.1f}%)")
        
        return self.features_df, self.labels
    
    def prepare_training_data(self):
        """Prepare data for TVAE training with robust normalization"""
        print("Preparing data for TVAE training...")
        
        # More robust normalization
        features_clipped = self.features_df.copy()
        for col in features_clipped.columns:
            q1 = features_clipped[col].quantile(0.01)
            q99 = features_clipped[col].quantile(0.99)
            features_clipped[col] = features_clipped[col].clip(q1, q99)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features_clipped.values)
        
        # Convert to tensors
        self.features_tensor = torch.FloatTensor(features_normalized).to(self.device)
        self.labels_tensor = torch.LongTensor(self.labels).to(self.device)
        self.conditions_tensor = torch.nn.functional.one_hot(self.labels_tensor, num_classes=2).float()
        
        self.feature_dim = self.features_tensor.shape[1]
        
        print(f"Data prepared: {self.features_tensor.shape}")
        print(f"Feature range after normalization: [{self.features_tensor.min():.3f}, {self.features_tensor.max():.3f}]")
        
        return self.features_tensor, self.conditions_tensor
    
    def initialize_tvae(self, latent_dim=32):
        """Initialize TVAE model"""
        print("Initializing TVAE model...")
        
        self.tvae = MoldingQualityTVAE(
            input_dim=self.feature_dim,
            condition_dim=2,
            latent_dim=latent_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.tvae.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        print(f"TVAE parameters: {sum(p.numel() for p in self.tvae.parameters())}")
        print(f"Latent dimension: {latent_dim}")
    
    def tvae_loss_function(self, reconstructed, original, mu, logvar, beta=1.0):
        """TVAE loss function combining reconstruction and KL divergence"""
        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss()(reconstructed, original)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / original.size(0)  # Normalize by batch size
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_tvae(self, epochs=1000, batch_size=32, print_interval=100, beta_schedule=True):
        """Train the TVAE model"""
        print(f"Starting TVAE training for {epochs} epochs...")
        
        dataset = TensorDataset(self.features_tensor, self.conditions_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training history
        total_losses = []
        recon_losses = []
        kl_losses = []
        
        for epoch in range(epochs):
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            batches = 0
            
            # Beta annealing for KL loss (starts low, increases gradually)
            if beta_schedule:
                beta = min(1.0, epoch / (epochs * 0.5))  # Reach full beta at 50% of training
            else:
                beta = 1.0
            
            for features, conditions in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                reconstructed, mu, logvar = self.tvae(features, conditions)
                
                # Calculate loss
                total_loss, recon_loss, kl_loss = self.tvae_loss_function(
                    reconstructed, features, mu, logvar, beta
                )
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                # Accumulate losses
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                batches += 1
            
            # Average losses
            avg_total_loss = epoch_total_loss / batches
            avg_recon_loss = epoch_recon_loss / batches
            avg_kl_loss = epoch_kl_loss / batches
            
            total_losses.append(avg_total_loss)
            recon_losses.append(avg_recon_loss)
            kl_losses.append(avg_kl_loss)
            
            # Print progress
            if (epoch + 1) % print_interval == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Total Loss: {avg_total_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, Beta: {beta:.3f}')
        
        print("TVAE training completed!")
        return total_losses, recon_losses, kl_losses
    
    def generate_synthetic_data_batched(self, n_acceptable=400, n_non_acceptable=200, batch_size=50):
        """Generate synthetic data in batches to avoid memory issues - FIXED VERSION"""
        print(f"Generating {n_acceptable} acceptable and {n_non_acceptable} non-acceptable samples with TVAE (batched)...")
        
        self.tvae.eval()
        synthetic_features_list = []
        synthetic_labels_list = []
        
        try:
            with torch.no_grad():
                # Generate acceptable samples in batches
                print("Generating acceptable samples in batches...")
                for i in range(0, n_acceptable, batch_size):
                    current_batch_size = min(batch_size, n_acceptable - i)
                    print(f"  Batch {i//batch_size + 1}: {current_batch_size} samples")
                    
                    # Create condition for acceptable class
                    condition = torch.zeros(current_batch_size, 2).to(self.device)
                    condition[:, 1] = 1  # Acceptable class
                    
                    # Sample from latent space
                    z = torch.randn(current_batch_size, self.tvae.latent_dim).to(self.device)
                    
                    # Generate features
                    generated = self.tvae.decoder(z, condition)
                    
                    # Move to CPU and store
                    batch_features = generated.cpu().numpy()
                    synthetic_features_list.append(batch_features)
                    synthetic_labels_list.extend([1] * current_batch_size)
                
                # Generate non-acceptable samples in batches
                print("Generating non-acceptable samples in batches...")
                for i in range(0, n_non_acceptable, batch_size):
                    current_batch_size = min(batch_size, n_non_acceptable - i)
                    print(f"  Batch {i//batch_size + 1}: {current_batch_size} samples")
                    
                    # Create condition for non-acceptable class
                    condition = torch.zeros(current_batch_size, 2).to(self.device)
                    condition[:, 0] = 1  # Non-acceptable class
                    
                    # Sample from latent space
                    z = torch.randn(current_batch_size, self.tvae.latent_dim).to(self.device)
                    
                    # Generate features
                    generated = self.tvae.decoder(z, condition)
                    
                    # Move to CPU and store
                    batch_features = generated.cpu().numpy()
                    synthetic_features_list.append(batch_features)
                    synthetic_labels_list.extend([0] * current_batch_size)
            
            # Combine all batches
            print("Combining all batches...")
            all_synthetic_features = np.vstack(synthetic_features_list)
            synthetic_labels = np.array(synthetic_labels_list)
            
            # Denormalize in chunks to avoid memory issues
            print("Denormalizing features...")
            chunk_size = 100
            denormalized_chunks = []
            
            for i in range(0, len(all_synthetic_features), chunk_size):
                chunk = all_synthetic_features[i:i+chunk_size]
                denorm_chunk = self.scaler.inverse_transform(chunk)
                denormalized_chunks.append(denorm_chunk)
            
            synthetic_features_denorm = np.vstack(denormalized_chunks)
            
            print(f"Successfully generated {len(synthetic_labels)} synthetic samples")
            print(f"Class distribution: Acceptable={np.sum(synthetic_labels)} ({np.sum(synthetic_labels)/len(synthetic_labels)*100:.1f}%), Non-acceptable={len(synthetic_labels) - np.sum(synthetic_labels)} ({(1-np.sum(synthetic_labels)/len(synthetic_labels))*100:.1f}%)")
            
            return synthetic_features_denorm, synthetic_labels
            
        except Exception as e:
            print(f"Error during batch generation: {e}")
            print("Trying with smaller batch size...")
            if batch_size > 20:
                return self.generate_synthetic_data_batched(n_acceptable, n_non_acceptable, batch_size=20)
            else:
                raise e
    
    def evaluate_tvae_quality(self, synthetic_features, synthetic_labels):
        """Evaluate the quality of TVAE synthetic data"""
        print("\n=== TVAE SYNTHETIC DATA QUALITY EVALUATION ===")
        
        synthetic_df = pd.DataFrame(synthetic_features, columns=self.features_df.columns)
        
        # Focus on critical parameters
        critical_params = ['Spritzdruck_mean', 'IST-Schliesskraft_mean', 'EE-Position_mean']
        
        quality_scores = []
        
        for param in critical_params:
            if param in self.features_df.columns:
                real_mean = self.features_df[param].mean()
                real_std = self.features_df[param].std()
                synthetic_mean = synthetic_df[param].mean()
                synthetic_std = synthetic_df[param].std()
                
                # Calculate quality score
                mean_diff = abs(real_mean - synthetic_mean) / abs(real_mean) if real_mean != 0 else 0
                std_diff = abs(real_std - synthetic_std) / real_std if real_std != 0 else 0
                quality_score = (mean_diff + std_diff) / 2
                quality_scores.append(quality_score)
                
                print(f"{param}:")
                print(f"  Real: μ={real_mean:.3f}, σ={real_std:.3f}")
                print(f"  Synthetic: μ={synthetic_mean:.3f}, σ={synthetic_std:.3f}")
                print(f"  Quality Score: {quality_score:.3f}")
        
        overall_quality = np.mean(quality_scores)
        print(f"\nOverall TVAE Quality Score: {overall_quality:.3f}")
        
        if overall_quality < 0.2:
            print("✅ EXCELLENT TVAE synthetic data quality!")
        elif overall_quality < 0.5:
            print("✅ GOOD TVAE synthetic data quality!")
        else:
            print("⚠️  ACCEPTABLE TVAE synthetic data quality")
        
        return overall_quality
    
    def create_augmented_dataset(self, synthetic_features, synthetic_labels):
        """Create final augmented dataset"""
        augmented_features = np.vstack([self.features_df.values, synthetic_features])
        augmented_labels = np.concatenate([self.labels, synthetic_labels])
        
        print(f"\n=== TVAE AUGMENTED DATASET ===")
        print(f"Original: {len(self.labels)} samples")
        print(f"TVAE Synthetic: {len(synthetic_labels)} samples")
        print(f"Total: {len(augmented_labels)} samples")
        print(f"Final class balance: Acceptable={np.sum(augmented_labels)} ({np.sum(augmented_labels)/len(augmented_labels)*100:.1f}%), Non-acceptable={len(augmented_labels) - np.sum(augmented_labels)} ({(1-np.sum(augmented_labels)/len(augmented_labels))*100:.1f}%)")
        
        return augmented_features, augmented_labels

# Main execution
if __name__ == "__main__":
    # Your folder paths
    acceptable_folder = "C:/Users/tc/Thesis_Project/Data_Sabic576P/io/io"
    non_acceptable_folder = "C:/Users/tc/Thesis_Project/Data_Sabic576P/nio/nio"
    
    try:
        # Initialize TVAE
        tvae = CompleteMoldingTVAE(acceptable_folder, non_acceptable_folder)
        
        # Load and prepare data
        features_df, labels = tvae.load_and_extract_features()
        features_tensor, conditions_tensor = tvae.prepare_training_data()
        
        # Initialize TVAE model
        tvae.initialize_tvae(latent_dim=32)
        
        # Train TVAE
        total_losses, recon_losses, kl_losses = tvae.train_tvae(
            epochs=1000, batch_size=32, print_interval=100
        )
        
        # Generate synthetic data using FIXED batched method
        synthetic_features, synthetic_labels = tvae.generate_synthetic_data_batched(
            n_acceptable=400, n_non_acceptable=200, batch_size=50
        )
        
        # Evaluate quality
        quality_score = tvae.evaluate_tvae_quality(synthetic_features, synthetic_labels)
        
        # Create augmented dataset
        augmented_features, augmented_labels = tvae.create_augmented_dataset(
            synthetic_features, synthetic_labels
        )
        
        print("\n🎉 TVAE TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print("Ready to retrain LSTM classifier with TVAE augmented dataset!")
        
        # Save results
        np.save('tvae_augmented_features.npy', augmented_features)
        np.save('tvae_augmented_labels.npy', augmented_labels)
        print("💾 TVAE augmented dataset saved to .npy files!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your folder paths and ensure CSV files are accessible.")