import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CycleFeatureExtractor:
    """Extract statistical and temporal features from molding cycles"""
    
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

class LightweightTransformerGenerator(nn.Module):
    """Memory-efficient Transformer for feature generation"""
    
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(LightweightTransformerGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim + 2, d_model)  # +2 for condition
        
        # Small transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,  # Much smaller
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        self.activation = nn.Tanh()
        
    def forward(self, x, condition):
        batch_size, feature_dim = x.shape
        
        # Treat each feature as a timestep in sequence
        x_reshaped = x.unsqueeze(2)  # (batch, feature_dim, 1)
        
        # Expand condition to match feature dimension
        condition_expanded = condition.unsqueeze(1).repeat(1, feature_dim, 1)
        
        # Concatenate input with condition
        x_with_condition = torch.cat([x_reshaped, condition_expanded], dim=2)
        
        # Project to model dimension
        projected = self.input_projection(x_with_condition)
        
        # Pass through transformer
        transformed = self.transformer(projected)
        
        # Project back to feature space
        output = self.output_projection(transformed)
        
        # Take mean across sequence dimension to get final feature vector
        output = torch.mean(output, dim=1).squeeze()
        
        return self.activation(output)

class SimplifiedTransformerAugmentation:
    """Simplified Transformer approach that works with statistical features directly"""
    
    def __init__(self, acceptable_folder, non_acceptable_folder):
        self.acceptable_folder = acceptable_folder
        self.non_acceptable_folder = non_acceptable_folder
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = CycleFeatureExtractor()
        
        print(f"Using device: {self.device}")
    
    def load_and_extract_features(self):
        """Load all data and extract statistical features (same as other methods)"""
        print("Loading and extracting features for Lightweight Transformer...")
        
        features_list = []
        labels_list = []
        
        # Process acceptable files (limit for memory)
        print("Processing acceptable files...")
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
        print("Processing non-acceptable files...")
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
    
    def prepare_data(self):
        """Prepare data for lightweight Transformer training"""
        print("Preparing data for Lightweight Transformer...")
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(self.features_df.values)
        
        # Convert to tensors
        self.features_tensor = torch.FloatTensor(features_normalized).to(self.device)
        self.labels_tensor = torch.LongTensor(self.labels).to(self.device)
        self.conditions_tensor = torch.nn.functional.one_hot(self.labels_tensor, num_classes=2).float()
        
        self.feature_dim = self.features_tensor.shape[1]
        
        print(f"Data prepared: {self.features_tensor.shape}")
        return self.features_tensor, self.conditions_tensor
    
    def initialize_transformer(self):
        """Initialize lightweight Transformer"""
        print("Initializing Lightweight Transformer...")
        
        self.transformer = LightweightTransformerGenerator(
            input_dim=self.feature_dim,
            d_model=64,
            nhead=4,
            num_layers=2
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        print(f"Transformer parameters: {sum(p.numel() for p in self.transformer.parameters())}")
    
    def train_transformer(self, epochs=1000, batch_size=32, print_interval=100):
        """Train the lightweight Transformer"""
        print(f"Starting Lightweight Transformer training for {epochs} epochs...")
        
        dataset = TensorDataset(self.features_tensor, self.conditions_tensor, self.features_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            batches = 0
            
            for features, conditions, targets in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass - reconstruct features
                output = self.transformer(features, conditions)
                
                # Reconstruction loss
                loss = self.criterion(output, targets)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1
            
            avg_loss = epoch_loss / batches
            losses.append(avg_loss)
            
            if (epoch + 1) % print_interval == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        print("Lightweight Transformer training completed!")
        return losses
    
    def generate_synthetic_features(self, n_acceptable=400, n_non_acceptable=200):
        """Generate synthetic features using trained Transformer"""
        print(f"Generating {n_acceptable} acceptable and {n_non_acceptable} non-acceptable samples...")
        
        self.transformer.eval()
        synthetic_features_list = []
        synthetic_labels_list = []
        
        with torch.no_grad():
            # Generate acceptable samples
            for _ in range(n_acceptable):
                condition = torch.FloatTensor([[0, 1]]).to(self.device)  # Acceptable
                
                # Use noise as input
                noise_input = torch.randn(1, self.feature_dim).to(self.device)
                
                # Generate features
                generated = self.transformer(noise_input, condition)
                
                synthetic_features_list.append(generated.cpu().numpy()[0])
                synthetic_labels_list.append(1)
            
            # Generate non-acceptable samples
            for _ in range(n_non_acceptable):
                condition = torch.FloatTensor([[1, 0]]).to(self.device)  # Non-acceptable
                
                noise_input = torch.randn(1, self.feature_dim).to(self.device)
                generated = self.transformer(noise_input, condition)
                
                synthetic_features_list.append(generated.cpu().numpy()[0])
                synthetic_labels_list.append(0)
        
        # Convert to arrays and denormalize
        synthetic_features = np.array(synthetic_features_list)
        synthetic_features_denorm = self.scaler.inverse_transform(synthetic_features)
        synthetic_labels = np.array(synthetic_labels_list)
        
        print(f"Generated {len(synthetic_labels)} synthetic samples")
        return synthetic_features_denorm, synthetic_labels
    
    def evaluate_quality(self, synthetic_features, synthetic_labels):
        """Evaluate synthetic data quality"""
        print("\n=== TRANSFORMER SYNTHETIC DATA QUALITY EVALUATION ===")
        
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
        print(f"\nOverall Transformer Quality Score: {overall_quality:.3f}")
        
        if overall_quality < 0.2:
            print("✅ EXCELLENT Transformer synthetic data quality!")
        elif overall_quality < 0.5:
            print("✅ GOOD Transformer synthetic data quality!")
        else:
            print("⚠️  ACCEPTABLE Transformer synthetic data quality")
        
        return overall_quality
    
    def create_augmented_dataset(self, synthetic_features, synthetic_labels):
        """Create final augmented dataset"""
        augmented_features = np.vstack([self.features_df.values, synthetic_features])
        augmented_labels = np.concatenate([self.labels, synthetic_labels])
        
        print(f"\n=== TRANSFORMER AUGMENTED DATASET ===")
        print(f"Original: {len(self.labels)} samples")
        print(f"Synthetic: {len(synthetic_labels)} samples")
        print(f"Total: {len(augmented_labels)} samples")
        print(f"Final class balance: Acceptable={np.sum(augmented_labels)} ({np.sum(augmented_labels)/len(augmented_labels)*100:.1f}%), Non-acceptable={len(augmented_labels) - np.sum(augmented_labels)} ({(1-np.sum(augmented_labels)/len(augmented_labels))*100:.1f}%)")
        
        return augmented_features, augmented_labels

# Main execution
if __name__ == "__main__":
    acceptable_folder = "C:/Users/tc/Thesis_Project/Data_Sabic576P/io/io"
    non_acceptable_folder = "C:/Users/tc/Thesis_Project/Data_Sabic576P/nio/nio"
    
    try:
        # Initialize lightweight Transformer
        transformer = SimplifiedTransformerAugmentation(acceptable_folder, non_acceptable_folder)
        
        # Load and prepare data
        features_df, labels = transformer.load_and_extract_features()
        features_tensor, conditions_tensor = transformer.prepare_data()
        
        # Initialize model
        transformer.initialize_transformer()
        
        # Train Transformer
        losses = transformer.train_transformer(epochs=1000, batch_size=32)
        
        # Generate synthetic data
        synthetic_features, synthetic_labels = transformer.generate_synthetic_features(
            n_acceptable=400, n_non_acceptable=200
        )
        
        # Evaluate quality
        quality_score = transformer.evaluate_quality(synthetic_features, synthetic_labels)
        
        # Create augmented dataset
        augmented_features, augmented_labels = transformer.create_augmented_dataset(
            synthetic_features, synthetic_labels
        )
        
        print("🎉 LIGHTWEIGHT TRANSFORMER COMPLETED SUCCESSFULLY! 🎉")
        
        # Save results
        np.save('transformer_augmented_features.npy', augmented_features)
        np.save('transformer_augmented_labels.npy', augmented_labels)
        print("💾 Transformer augmented dataset saved!")
        
    except Exception as e:
        print(f"❌ Error: {e}")