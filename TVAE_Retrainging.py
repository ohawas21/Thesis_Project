import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalLSTMClassifier(nn.Module):
    """LSTM classifier for statistical features (same architecture as baseline)"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(StatisticalLSTMClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Dense layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 1)
        
        lstm_out, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

class TVAELSTMTrainer:
    """Train LSTM on TVAE augmented dataset and compare with CT-GAN results"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.scaler = StandardScaler()
    
    def load_tvae_data(self):
        """Load TVAE augmented dataset"""
        print("Loading TVAE augmented dataset...")
        
        try:
            self.features = np.load('tvae_augmented_features.npy')
            self.labels = np.load('tvae_augmented_labels.npy')
            
            print(f"Loaded TVAE augmented dataset: {self.features.shape}")
            print(f"Class distribution: Acceptable={np.sum(self.labels)} ({np.sum(self.labels)/len(self.labels)*100:.1f}%), Non-acceptable={len(self.labels) - np.sum(self.labels)} ({(1-np.sum(self.labels)/len(self.labels))*100:.1f}%)")
            
            return self.features, self.labels
            
        except FileNotFoundError:
            print("❌ TVAE augmented files not found!")
            print("Make sure 'tvae_augmented_features.npy' and 'tvae_augmented_labels.npy' exist.")
            raise
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare TVAE augmented data for training"""
        print("Preparing TVAE augmented training and testing data...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train_scaled).to(self.device)
        self.X_test = torch.FloatTensor(X_test_scaled).to(self.device)
        self.y_train = torch.LongTensor(y_train).to(self.device)
        self.y_test = torch.LongTensor(y_test).to(self.device)
        
        print(f"TVAE training set: {self.X_train.shape[0]} samples")
        print(f"TVAE testing set: {self.X_test.shape[0]} samples")
        print(f"Feature dimensions: {self.X_train.shape[1]}")
        
        # Class distribution in train/test
        train_acceptable = torch.sum(self.y_train).item()
        test_acceptable = torch.sum(self.y_test).item()
        print(f"Training set - Acceptable: {train_acceptable} ({train_acceptable/len(self.y_train)*100:.1f}%)")
        print(f"Testing set - Acceptable: {test_acceptable} ({test_acceptable/len(self.y_test)*100:.1f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_tvae_model(self, epochs=150, batch_size=64, learning_rate=0.001):
        """Train LSTM on TVAE augmented dataset"""
        print("\n" + "="*60)
        print("TRAINING LSTM ON TVAE AUGMENTED DATASET")
        print("="*60)
        
        # Initialize model
        input_size = self.X_train.shape[1]
        self.model = StatisticalLSTMClassifier(input_size).to(self.device)
        
        # Class weights for any remaining imbalance
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
        
        print(f"Starting TVAE-LSTM training for {epochs} epochs...")
        
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
        
        print("TVAE-LSTM training completed!")
        return train_losses, train_accuracies
    
    def evaluate_tvae_model(self):
        """Evaluate the TVAE-trained model"""
        print("\n" + "="*60)
        print("TVAE AUGMENTED MODEL EVALUATION RESULTS")
        print("="*60)
        
        self.model.eval()
        with torch.no_grad():
            # Predictions on test set
            test_outputs = self.model(self.X_test)
            _, test_predicted = torch.max(test_outputs.data, 1)
            
            # Convert to numpy for sklearn metrics
            y_true = self.y_test.cpu().numpy()
            y_pred = test_predicted.cpu().numpy()
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            # Class-specific metrics
            precision_per_class = precision_score(y_true, y_pred, average=None)
            recall_per_class = recall_score(y_true, y_pred, average=None)
            f1_per_class = f1_score(y_true, y_pred, average=None)
            
            print(f"Overall Accuracy: {accuracy:.4f}")
            print(f"Overall Precision: {precision:.4f}")
            print(f"Overall Recall: {recall:.4f}")
            print(f"Overall F1-Score: {f1:.4f}")
            
            print(f"\nClass-specific Performance:")
            print(f"Non-Acceptable - Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}")
            print(f"Acceptable - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}")
            
            print("\nDetailed Classification Report:")
            print(classification_report(y_true, y_pred, target_names=['Non-Acceptable', 'Acceptable']))
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Non-Acceptable', 'Acceptable'],
                       yticklabels=['Non-Acceptable', 'Acceptable'])
            plt.title('Confusion Matrix - TVAE Augmented LSTM Classifier')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
            
            return accuracy, precision, recall, f1, precision_per_class, recall_per_class
    
    def compare_all_methods(self):
        """Compare TVAE results with baseline and CT-GAN"""
        print("\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE COMPARISON")
        print("="*70)
        
        # Get TVAE results
        tvae_accuracy, tvae_precision, tvae_recall, tvae_f1, tvae_prec_class, tvae_rec_class = self.evaluate_tvae_model()
        
        # Known results from previous experiments
        baseline_results = {
            'accuracy': 0.7192,
            'acceptable_precision': 0.58,
            'acceptable_recall': 0.59
        }
        
        ct_gan_results = {
            'accuracy': 0.7594,
            'acceptable_precision': 0.7778,
            'acceptable_recall': 0.7054
        }
        
        # Create comparison table
        print(f"\n📊 OVERALL PERFORMANCE COMPARISON:")
        print(f"{'Method':<15} {'Accuracy':<10} {'Acc. Precision':<15} {'Acc. Recall':<12} {'Improvement'}")
        print("-" * 65)
        baseline_acc = baseline_results['accuracy']
        baseline_prec = baseline_results['acceptable_precision']
        baseline_rec = baseline_results['acceptable_recall']
        ctgan_acc = ct_gan_results['accuracy']
        ctgan_prec = ct_gan_results['acceptable_precision']
        ctgan_rec = ct_gan_results['acceptable_recall']
        
        tvae_improvement = (tvae_prec_class[1] - baseline_prec) * 100
        
        print(f"{'Baseline':<15} {baseline_acc:<10.4f} {baseline_prec:<15.4f} {baseline_rec:<12.4f} {'---'}")
        print(f"{'CT-GAN':<15} {ctgan_acc:<10.4f} {ctgan_prec:<15.4f} {ctgan_rec:<12.4f} {'+19.8pp'}")
        print(f"{'TVAE':<15} {tvae_accuracy:<10.4f} {tvae_prec_class[1]:<15.4f} {tvae_rec_class[1]:<12.4f} {f'+{tvae_improvement:.1f}pp'}")
        
        print(f"\n🎯 KEY INSIGHTS:")
        
        # TVAE vs Baseline comparison
        tvae_acc_improvement = (tvae_accuracy - baseline_results['accuracy']) * 100
        tvae_prec_improvement = (tvae_prec_class[1] - baseline_results['acceptable_precision']) * 100
        tvae_rec_improvement = (tvae_rec_class[1] - baseline_results['acceptable_recall']) * 100
        
        print(f"TVAE vs Baseline:")
        print(f"  • Accuracy: +{tvae_acc_improvement:.1f} percentage points")
        print(f"  • Acceptable Precision: +{tvae_prec_improvement:.1f} percentage points")
        print(f"  • Acceptable Recall: +{tvae_rec_improvement:.1f} percentage points")
        
        # TVAE vs CT-GAN comparison
        tvae_vs_ctgan_acc = (tvae_accuracy - ct_gan_results['accuracy']) * 100
        tvae_vs_ctgan_prec = (tvae_prec_class[1] - ct_gan_results['acceptable_precision']) * 100
        tvae_vs_ctgan_rec = (tvae_rec_class[1] - ct_gan_results['acceptable_recall']) * 100
        
        print(f"\nTVAE vs CT-GAN:")
        print(f"  • Accuracy: {tvae_vs_ctgan_acc:+.1f} percentage points")
        print(f"  • Acceptable Precision: {tvae_vs_ctgan_prec:+.1f} percentage points")
        print(f"  • Acceptable Recall: {tvae_vs_ctgan_rec:+.1f} percentage points")
        
        # Business impact
        baseline_waste = (1 - baseline_results['acceptable_recall']) * 100
        ctgan_waste = (1 - ct_gan_results['acceptable_recall']) * 100
        tvae_waste = (1 - tvae_rec_class[1]) * 100
        
        print(f"\n💼 BUSINESS IMPACT (Good Parts Rejected):")
        print(f"  • Baseline: {baseline_waste:.1f}% of good parts rejected")
        print(f"  • CT-GAN: {ctgan_waste:.1f}% of good parts rejected")
        print(f"  • TVAE: {tvae_waste:.1f}% of good parts rejected")
        
        # Winner determination
        if tvae_prec_class[1] > ct_gan_results['acceptable_precision']:
            print(f"\n🏆 WINNER: TVAE performs better on acceptable precision!")
        elif tvae_prec_class[1] < ct_gan_results['acceptable_precision']:
            print(f"\n🏆 WINNER: CT-GAN performs better on acceptable precision!")
        else:
            print(f"\n🤝 TIE: Both methods perform similarly!")
        
        # Quality vs Performance analysis
        print(f"\n🔬 SYNTHETIC DATA QUALITY vs PERFORMANCE:")
        print(f"  • CT-GAN Quality Score: 0.125 (Excellent) → Precision: {ct_gan_results['acceptable_precision']:.3f}")
        print(f"  • TVAE Quality Score: 0.447 (Good) → Precision: {tvae_prec_class[1]:.3f}")
        print(f"  • Observation: Lower quality score doesn't always mean worse performance!")
        
        return {
            'accuracy': tvae_accuracy,
            'precision': tvae_precision,
            'recall': tvae_recall,
            'f1': tvae_f1,
            'acceptable_precision': tvae_prec_class[1],
            'acceptable_recall': tvae_rec_class[1]
        }

# Main execution
if __name__ == "__main__":
    # Initialize TVAE LSTM trainer
    trainer = TVAELSTMTrainer()
    
    # Load TVAE augmented data
    features, labels = trainer.load_tvae_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data()
    
    # Train model on TVAE augmented data
    train_losses, train_accuracies = trainer.train_tvae_model(epochs=150)
    
    # Comprehensive comparison
    tvae_results = trainer.compare_all_methods()
    
    print("\n🎉 TVAE EVALUATION COMPLETED! 🎉")
    print("Now we have results for Baseline, CT-GAN, and TVAE!")
    print("Ready for Transformer implementation next!")