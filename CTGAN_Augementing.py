import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class CycleFeatureExtractor:
    """Extract statistical and temporal features from molding cycles"""
    
    def __init__(self, critical_params=None):
        if critical_params is None:
            # Based on your domain knowledge - Priority 1 & 2 parameters
            self.critical_params = [
                # Priority 1
                'Spritzdruck', 'IST-Schliesskraft', 'EE-Position',
                # Priority 2  
                'SC-Position', 'SC-Drehzahl', 'SC-Drehmoment', 'Anlagekraft',
                # Additional important ones
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
                    # Linear trend slope
                    slope, _ = np.polyfit(range(len(values)), values, 1)
                    features[f'{param}_trend'] = slope
                    
                    # First and last values
                    features[f'{param}_start'] = values[0]
                    features[f'{param}_end'] = values[-1]
                    features[f'{param}_change'] = values[-1] - values[0]
                    
                    # Variability measures
                    features[f'{param}_cv'] = np.std(values) / np.abs(np.mean(values)) if np.mean(values) != 0 else 0
                    
                    # Rate of change (derivative approximation)
                    diff_values = np.diff(values)
                    features[f'{param}_mean_derivative'] = np.mean(diff_values)
                    features[f'{param}_std_derivative'] = np.std(diff_values)
        
        # Cycle-level features
        features['cycle_length'] = len(cycle_data)
        features['cycle_duration_ratio'] = len(cycle_data) / 3000  # Normalized to typical length
        
        return features

class ImprovedCTGANGenerator(nn.Module):
    """Improved CT-GAN Generator with better stability"""
    
    def __init__(self, noise_dim, condition_dim, feature_dim, hidden_dims=[128, 256, 128]):
        super(ImprovedCTGANGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.feature_dim = feature_dim
        
        input_dim = noise_dim + condition_dim
        
        # Smaller, more stable architecture
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),  # LeakyReLU instead of ReLU
                nn.Dropout(0.1)     # Lower dropout
            ])
            prev_dim = hidden_dim
        
        # Output layer - no activation to allow full range
        layers.append(nn.Linear(prev_dim, feature_dim))
        
        self.generator = nn.Sequential(*layers)
        
    def forward(self, noise, condition):
        gen_input = torch.cat([noise, condition], dim=1)
        return self.generator(gen_input)

class ImprovedCTGANDiscriminator(nn.Module):
    """Improved CT-GAN Discriminator with better stability"""
    
    def __init__(self, feature_dim, condition_dim, hidden_dims=[128, 64]):
        super(ImprovedCTGANDiscriminator, self).__init__()
        
        input_dim = feature_dim + condition_dim
        
        # Smaller discriminator to prevent overpowering generator
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.discriminator = nn.Sequential(*layers)
        
    def forward(self, features, condition):
        disc_input = torch.cat([features, condition], dim=1)
        return self.discriminator(disc_input)

class CompleteMoldingCTGAN:
    """Complete standalone CT-GAN implementation for molding quality data"""
    
    def __init__(self, acceptable_folder, non_acceptable_folder):
        self.acceptable_folder = acceptable_folder
        self.non_acceptable_folder = non_acceptable_folder
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = CycleFeatureExtractor()
        
        # Improved parameters
        self.noise_dim = 64
        self.condition_dim = 2
        
        print(f"Using device: {self.device}")
    
    def load_and_extract_features(self):
        """Load all data and extract statistical features"""
        print("Loading and extracting features for improved CT-GAN...")
        
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
        """Prepare data with improved normalization"""
        print("Preparing data for improved CT-GAN training...")
        
        # More robust normalization - clip outliers first
        features_clipped = self.features_df.copy()
        for col in features_clipped.columns:
            q1 = features_clipped[col].quantile(0.01)
            q99 = features_clipped[col].quantile(0.99)
            features_clipped[col] = features_clipped[col].clip(q1, q99)
        
        # Normalize
        features_normalized = self.scaler.fit_transform(features_clipped.values)
        
        # Convert to tensors
        self.features_tensor = torch.FloatTensor(features_normalized).to(self.device)
        self.labels_tensor = torch.LongTensor(self.labels).to(self.device)
        self.conditions_tensor = torch.nn.functional.one_hot(self.labels_tensor, num_classes=2).float()
        
        self.feature_dim = self.features_tensor.shape[1]
        
        print(f"Data prepared: {self.features_tensor.shape}")
        print(f"Feature range after normalization: [{self.features_tensor.min():.3f}, {self.features_tensor.max():.3f}]")
        
        return self.features_tensor, self.conditions_tensor
    
    def initialize_models(self):
        """Initialize improved models"""
        print("Initializing improved CT-GAN models...")
        
        self.generator = ImprovedCTGANGenerator(
            noise_dim=self.noise_dim,
            condition_dim=self.condition_dim,
            feature_dim=self.feature_dim
        ).to(self.device)
        
        self.discriminator = ImprovedCTGANDiscriminator(
            feature_dim=self.feature_dim,
            condition_dim=self.condition_dim
        ).to(self.device)
        
        # Improved optimizers with different learning rates
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters())}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters())}")
    
    def train_improved_ctgan(self, epochs=1500, batch_size=32, print_interval=150):
        """Improved training with better stability"""
        print(f"Starting improved CT-GAN training for {epochs} epochs...")
        
        dataset = TensorDataset(self.features_tensor, self.conditions_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        gen_losses = []
        disc_losses = []
        
        # Training parameters for stability
        real_label_smoothing = 0.9
        fake_label_smoothing = 0.1
        
        for epoch in range(epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            batches = 0
            
            for real_features, conditions in dataloader:
                batch_size_actual = real_features.size(0)
                
                # Smoothed labels for stability
                real_labels = torch.full((batch_size_actual, 1), real_label_smoothing).to(self.device)
                fake_labels = torch.full((batch_size_actual, 1), fake_label_smoothing).to(self.device)
                
                # Train Discriminator (less frequently)
                if epoch % 2 == 0:
                    self.disc_optimizer.zero_grad()
                    
                    # Real data
                    real_pred = self.discriminator(real_features, conditions)
                    real_loss = self.criterion(real_pred, real_labels)
                    
                    # Fake data
                    noise = torch.randn(batch_size_actual, self.noise_dim).to(self.device)
                    fake_features = self.generator(noise, conditions)
                    fake_pred = self.discriminator(fake_features.detach(), conditions)
                    fake_loss = self.criterion(fake_pred, fake_labels)
                    
                    disc_loss = real_loss + fake_loss
                    disc_loss.backward()
                    self.disc_optimizer.step()
                else:
                    disc_loss = torch.tensor(0.0)
                
                # Train Generator
                self.gen_optimizer.zero_grad()
                
                noise = torch.randn(batch_size_actual, self.noise_dim).to(self.device)
                fake_features = self.generator(noise, conditions)
                fake_pred = self.discriminator(fake_features, conditions)
                
                gen_target = torch.full((batch_size_actual, 1), real_label_smoothing).to(self.device)
                gen_loss = self.criterion(fake_pred, gen_target)
                
                gen_loss.backward()
                self.gen_optimizer.step()
                
                epoch_gen_loss += gen_loss.item()
                epoch_disc_loss += disc_loss.item() if isinstance(disc_loss, torch.Tensor) else 0
                batches += 1
            
            avg_gen_loss = epoch_gen_loss / batches
            avg_disc_loss = epoch_disc_loss / batches
            gen_losses.append(avg_gen_loss)
            disc_losses.append(avg_disc_loss)
            
            if (epoch + 1) % print_interval == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}')
        
        print("Improved CT-GAN training completed!")
        return gen_losses, disc_losses
    
    def generate_synthetic_data(self, n_acceptable=400, n_non_acceptable=200):
        """Generate synthetic data with better quality control"""
        print(f"Generating {n_acceptable} acceptable and {n_non_acceptable} non-acceptable samples...")
        
        self.generator.eval()
        
        with torch.no_grad():
            # Generate acceptable samples
            condition = torch.zeros(n_acceptable, 2).to(self.device)
            condition[:, 1] = 1
            noise = torch.randn(n_acceptable, self.noise_dim).to(self.device)
            acceptable_features = self.generator(noise, condition)
            
            # Generate non-acceptable samples
            condition = torch.zeros(n_non_acceptable, 2).to(self.device)
            condition[:, 0] = 1
            noise = torch.randn(n_non_acceptable, self.noise_dim).to(self.device)
            non_acceptable_features = self.generator(noise, condition)
            
            # Combine and denormalize
            all_synthetic = torch.cat([acceptable_features, non_acceptable_features], dim=0)
            synthetic_features_np = all_synthetic.cpu().numpy()
            synthetic_features_denorm = self.scaler.inverse_transform(synthetic_features_np)
            
            # Create labels
            synthetic_labels = np.concatenate([
                np.ones(n_acceptable),
                np.zeros(n_non_acceptable)
            ])
        
        print(f"Generated {len(synthetic_labels)} synthetic samples")
        return synthetic_features_denorm, synthetic_labels
    
    def evaluate_quality(self, synthetic_features, synthetic_labels):
        """Enhanced quality evaluation"""
        print("\n=== SYNTHETIC DATA QUALITY EVALUATION ===")
        
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
        print(f"\nOverall Quality Score: {overall_quality:.3f}")
        
        if overall_quality < 0.2:
            print("✅ EXCELLENT synthetic data quality!")
        elif overall_quality < 0.5:
            print("✅ GOOD synthetic data quality!")
        else:
            print("⚠️  ACCEPTABLE synthetic data quality")
        
        return overall_quality
    
    def create_augmented_dataset(self, synthetic_features, synthetic_labels):
        """Create final augmented dataset"""
        augmented_features = np.vstack([self.features_df.values, synthetic_features])
        augmented_labels = np.concatenate([self.labels, synthetic_labels])
        
        print(f"\n=== FINAL AUGMENTED DATASET ===")
        print(f"Original: {len(self.labels)} samples")
        print(f"Synthetic: {len(synthetic_labels)} samples")
        print(f"Total: {len(augmented_labels)} samples")
        print(f"Final class balance: Acceptable={np.sum(augmented_labels)} ({np.sum(augmented_labels)/len(augmented_labels)*100:.1f}%), Non-acceptable={len(augmented_labels) - np.sum(augmented_labels)} ({(1-np.sum(augmented_labels)/len(augmented_labels))*100:.1f}%)")
        
        return augmented_features, augmented_labels

# Main execution
if __name__ == "__main__":
    # Your folder paths
    acceptable_folder = "C:/Users/tc/Thesis_Project/Data_Sabic576P/io/io"
    non_acceptable_folder = "C:/Users/tc/Thesis_Project/Data_Sabic576P/nio/nio"
    
    try:
        # Initialize
        ctgan = CompleteMoldingCTGAN(acceptable_folder, non_acceptable_folder)
        
        # Load and prepare data
        features_df, labels = ctgan.load_and_extract_features()
        features_tensor, conditions_tensor = ctgan.prepare_training_data()
        
        # Initialize models
        ctgan.initialize_models()
        
        # Train CT-GAN
        gen_losses, disc_losses = ctgan.train_improved_ctgan(epochs=1500, batch_size=32)
        
        # Generate synthetic data
        synthetic_features, synthetic_labels = ctgan.generate_synthetic_data(
            n_acceptable=400, n_non_acceptable=200
        )
        
        # Evaluate quality
        quality_score = ctgan.evaluate_quality(synthetic_features, synthetic_labels)
        
        # Create augmented dataset
        augmented_features, augmented_labels = ctgan.create_augmented_dataset(
            synthetic_features, synthetic_labels
        )
        
        print("\n🎉 IMPROVED CT-GAN COMPLETED SUCCESSFULLY! 🎉")
        print("Ready to retrain your LSTM classifier with the augmented dataset!")
        
        # Save results (optional)
        np.save('augmented_features.npy', augmented_features)
        np.save('augmented_labels.npy', augmented_labels)
        print("💾 Augmented dataset saved to .npy files!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your folder paths and ensure CSV files are accessible.")