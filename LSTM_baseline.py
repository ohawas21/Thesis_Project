import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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

class StatisticalLSTMClassifier(nn.Module):
    """LSTM classifier for statistical features"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(StatisticalLSTMClassifier, self).__init__()
        
        # Since we're using statistical features, we'll reshape them for LSTM
        # We'll group features by parameter for sequence-like processing
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=1,  # We'll feed features one at a time
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Dense layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Binary classification
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape features for LSTM: (batch_size, seq_len, 1)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 1)  # Each feature becomes a time step
        
        # LSTM forward pass
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use the last hidden state
        out = self.dropout(hidden[-1])  # Take last layer's hidden state
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

class MoldingQualityClassifier:
    """Complete pipeline for molding quality classification"""
    
    def __init__(self, acceptable_folder, non_acceptable_folder):
        self.acceptable_folder = acceptable_folder
        self.non_acceptable_folder = non_acceptable_folder
        self.feature_extractor = CycleFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        
    def load_and_extract_features(self):
        """Load all data and extract statistical features"""
        print("Loading and extracting features from all files...")
        
        features_list = []
        labels_list = []
        
        # Process acceptable files
        print(f"Processing acceptable files from: {self.acceptable_folder}")
        acceptable_files = [f for f in os.listdir(self.acceptable_folder) if f.endswith('.csv')]
        
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
        
        # Process non-acceptable files
        print(f"Processing non-acceptable files from: {self.non_acceptable_folder}")
        non_acceptable_files = [f for f in os.listdir(self.non_acceptable_folder) if f.endswith('.csv')]
        
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
        
        # Convert to DataFrame
        self.features_df = pd.DataFrame(features_list)
        self.labels = np.array(labels_list)
        
        print(f"Extracted features from {len(features_list)} cycles")
        print(f"Feature dimensions: {self.features_df.shape}")
        print(f"Class distribution: Acceptable={np.sum(self.labels)} ({np.sum(self.labels)/len(self.labels)*100:.1f}%), Non-acceptable={len(self.labels) - np.sum(self.labels)} ({(1-np.sum(self.labels)/len(self.labels))*100:.1f}%)")
        
        # Feature importance preview
        print(f"\nExtracted {self.features_df.shape[1]} features per cycle")
        print("Sample features:", list(self.features_df.columns)[:10], "...")
        
        # Handle missing values
        self.features_df = self.features_df.fillna(0)
        
        return self.features_df, self.labels
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare training and testing data"""
        print("Preparing training and testing data...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features_df.values, self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels  # Maintain class balance
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train_scaled)
        self.X_test = torch.FloatTensor(X_test_scaled)
        self.y_train = torch.LongTensor(y_train)
        self.y_test = torch.LongTensor(y_test)
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        print(f"Feature dimensions: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, epochs=150, batch_size=64, learning_rate=0.001):
        """Train the LSTM classifier"""
        print("Training LSTM classifier...")
        
        # Initialize model
        input_size = self.X_train.shape[1]
        self.model = StatisticalLSTMClassifier(input_size)
        
        # Loss and optimizer with class weights for imbalanced data
        class_counts = torch.bincount(self.y_train)
        class_weights = len(self.y_train) / (2 * class_counts.float())
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            if (epoch + 1) % 30 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return train_losses, train_accuracies
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        self.model.eval()
        with torch.no_grad():
            # Predictions on test set
            test_outputs = self.model(self.X_test)
            _, test_predicted = torch.max(test_outputs.data, 1)
            
            # Convert to numpy for sklearn metrics
            y_true = self.y_test.numpy()
            y_pred = test_predicted.numpy()
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            print("\n" + "="*50)
            print("MODEL EVALUATION RESULTS")
            print("="*50)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print("\nDetailed Classification Report:")
            print(classification_report(y_true, y_pred, target_names=['Non-Acceptable', 'Acceptable']))
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-Acceptable', 'Acceptable'],
                       yticklabels=['Non-Acceptable', 'Acceptable'])
            plt.title('Confusion Matrix - Statistical Feature LSTM')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
            
            return accuracy, precision, recall, f1

# Usage example - you'll need to specify your actual folder paths
if __name__ == "__main__":
    # MODIFY THESE PATHS TO YOUR ACTUAL DATA FOLDERS
    acceptable_folder = "/Users/omarhawas/Thesis_Project/Data_Sabic576P/io/io"  # Folder with _io.csv files
    non_acceptable_folder = "/Users/omarhawas/Thesis_Project/Data_Sabic576P/nio/nio"  # Folder with _nio.csv files

    # Initialize classifier
    classifier = MoldingQualityClassifier(acceptable_folder, non_acceptable_folder)
    
    # Load data and extract features
    features, labels = classifier.load_and_extract_features()
    
    # Prepare training/testing data
    X_train, X_test, y_train, y_test = classifier.prepare_data()
    
    # Train model
    train_losses, train_accuracies = classifier.train_model(epochs=150)
    
    # Evaluate model
    accuracy, precision, recall, f1 = classifier.evaluate_model()
    
    print(f"\nFinal Results Summary:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")