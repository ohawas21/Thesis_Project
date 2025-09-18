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
    """LSTM classifier for statistical features (same as baseline)"""
    
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

class AugmentedLSTMTrainer:
    """Train LSTM on CT-GAN augmented dataset"""
    
    def __init__(self, augmented_features_path, augmented_labels_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load augmented data
        print("Loading CT-GAN augmented dataset...")
        self.features = np.load(augmented_features_path)
        self.labels = np.load(augmented_labels_path)
        
        print(f"Loaded augmented dataset: {self.features.shape}")
        print(f"Class distribution: Acceptable={np.sum(self.labels)} ({np.sum(self.labels)/len(self.labels)*100:.1f}%), Non-acceptable={len(self.labels) - np.sum(self.labels)} ({(1-np.sum(self.labels)/len(self.labels))*100:.1f}%)")
        
        self.scaler = StandardScaler()
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare augmented data for training"""
        print("Preparing augmented training and testing data...")
        
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
        
        print(f"Augmented training set: {self.X_train.shape[0]} samples")
        print(f"Augmented testing set: {self.X_test.shape[0]} samples")
        print(f"Feature dimensions: {self.X_train.shape[1]}")
        
        # Class distribution in train/test
        train_acceptable = torch.sum(self.y_train).item()
        test_acceptable = torch.sum(self.y_test).item()
        print(f"Training set - Acceptable: {train_acceptable} ({train_acceptable/len(self.y_train)*100:.1f}%)")
        print(f"Testing set - Acceptable: {test_acceptable} ({test_acceptable/len(self.y_test)*100:.1f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_augmented_model(self, epochs=150, batch_size=64, learning_rate=0.001):
        """Train LSTM on augmented dataset"""
        print("\n" + "="*60)
        print("TRAINING LSTM ON CT-GAN AUGMENTED DATASET")
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
        
        print(f"Starting training for {epochs} epochs...")
        
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
        
        print("Training completed!")
        return train_losses, train_accuracies
    
    def evaluate_augmented_model(self):
        """Evaluate the trained model on augmented data"""
        print("\n" + "="*60)
        print("AUGMENTED MODEL EVALUATION RESULTS")
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
            plt.title('Confusion Matrix - CT-GAN Augmented LSTM Classifier')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
            
            return accuracy, precision, recall, f1, precision_per_class, recall_per_class
    
    def compare_with_baseline(self, baseline_results):
        """Compare augmented results with baseline"""
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON: BASELINE vs CT-GAN AUGMENTED")
        print("="*60)
        
        # Run evaluation
        aug_accuracy, aug_precision, aug_recall, aug_f1, aug_prec_class, aug_rec_class = self.evaluate_augmented_model()
        
        # Baseline results (from your original run)
        baseline_accuracy = baseline_results.get('accuracy', 0.7192)
        baseline_acceptable_precision = baseline_results.get('acceptable_precision', 0.58)
        baseline_acceptable_recall = baseline_results.get('acceptable_recall', 0.59)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"Baseline Accuracy:     {baseline_accuracy:.4f}")
        print(f"Augmented Accuracy:    {aug_accuracy:.4f}")
        print(f"Improvement:           +{(aug_accuracy - baseline_accuracy):.4f} ({((aug_accuracy - baseline_accuracy)/baseline_accuracy)*100:+.1f}%)")
        
        print(f"\n🎯 ACCEPTABLE CLASS PERFORMANCE (Critical!):")
        print(f"Baseline Precision:    {baseline_acceptable_precision:.4f}")
        print(f"Augmented Precision:   {aug_prec_class[1]:.4f}")
        print(f"Improvement:           +{(aug_prec_class[1] - baseline_acceptable_precision):.4f} ({((aug_prec_class[1] - baseline_acceptable_precision)/baseline_acceptable_precision)*100:+.1f}%)")
        
        print(f"\nBaseline Recall:       {baseline_acceptable_recall:.4f}")
        print(f"Augmented Recall:      {aug_rec_class[1]:.4f}")
        print(f"Improvement:           +{(aug_rec_class[1] - baseline_acceptable_recall):.4f} ({((aug_rec_class[1] - baseline_acceptable_recall)/baseline_acceptable_recall)*100:+.1f}%)")
        
        # Business impact
        baseline_missed = (1 - baseline_acceptable_recall) * 100
        augmented_missed = (1 - aug_rec_class[1]) * 100
        print(f"\n💼 BUSINESS IMPACT:")
        print(f"Baseline: {baseline_missed:.1f}% of good parts incorrectly rejected")
        print(f"Augmented: {augmented_missed:.1f}% of good parts incorrectly rejected")
        print(f"Reduction: {baseline_missed - augmented_missed:.1f} percentage points fewer rejections")
        
        return {
            'accuracy': aug_accuracy,
            'precision': aug_precision,
            'recall': aug_recall,
            'f1': aug_f1,
            'acceptable_precision': aug_prec_class[1],
            'acceptable_recall': aug_rec_class[1]
        }

# Main execution
if __name__ == "__main__":
    # Load and train on augmented dataset
    trainer = AugmentedLSTMTrainer('augmented_features.npy', 'augmented_labels.npy')
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data()
    
    # Train model
    train_losses, train_accuracies = trainer.train_augmented_model(epochs=150)
    
    # Compare with baseline results
    baseline_results = {
        'accuracy': 0.7192,
        'acceptable_precision': 0.58,
        'acceptable_recall': 0.59
    }
    
    # Evaluate and compare
    augmented_results = trainer.compare_with_baseline(baseline_results)
    
    print("\n🎉 AUGMENTED TRAINING COMPLETED! 🎉")
    print("Check the results above to see the improvement from CT-GAN synthetic data!")