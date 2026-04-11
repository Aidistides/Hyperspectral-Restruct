"""
Baseline Models for Microplastics in Soil
Inspired by:
- Liu et al. (2023) Environ. Sci. Technol. - 1D CNN & RF on raw FT-IR spectra for blended MPs
- Li et al. (2021) Chemosphere - LS-SVM for rapid quantification of MPs (LDPE/PVC) in soil, with local vs. multisource modeling

This module provides simple, strong baselines (1D CNN, RF, LS-SVM) that can be compared against your main 3D CNN.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR  # or SVC for classification; here we use regression-style for concentration
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class Simple1DCNN(nn.Module):
    """Light 1D CNN baseline (inspired by Liu et al. 2023) – works on raw or preprocessed spectra."""
    def __init__(self, num_bands: int, num_classes: int = 8):  # adjust classes for your polymers
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * (num_bands // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, bands)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class MPBaselineModels:
    """
    Collection of baselines:
    - 1D CNN (raw spectra, high performance with enough data)
    - Random Forest (robust with limited data)
    - LS-SVM style regression (for quantification, multisource training)
    """
    
    def __init__(self):
        self.rf = None
        self.svm_reg = None
        self.cnn = None
    
    def train_rf(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Random Forest baseline – robust when data is limited (Liu et al.)."""
        if X.shape[0] < 10:
            raise ValueError(f"Insufficient training data: {X.shape[0]} samples (minimum 10 required)")
        
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D data (samples, features), got shape {X.shape}")
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        try:
            self.rf = RandomForestClassifier(
                n_estimators=200, max_depth=None, random_state=42, **kwargs
            )
            self.rf.fit(X_train, y_train)
            y_pred = self.rf.predict(X_test)
            
            print("=== Random Forest Baseline ===")
            print(classification_report(y_test, y_pred, zero_division=0))
            return self.rf
            
        except Exception as e:
            raise RuntimeError(f"Random Forest training failed: {e}")
    
    def train_1d_cnn(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, batch_size: int = 32, lr: float = 0.001):
        """1D CNN on raw spectra (strong performer per Liu et al. 2023)."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        num_bands = X.shape[1]
        num_classes = len(np.unique(y))
        self.cnn = Simple1DCNN(num_bands, num_classes)
        
        optimizer = optim.Adam(self.cnn.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.cnn.train()
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.cnn(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
        
        print("✅ 1D CNN trained (raw spectra baseline)")
        return self.cnn
    
    def train_multisource_svm(self, X_list: list, y_list: list, target_concentration: bool = True):
        """
        Multisource training (inspired by Li et al. 2021) – combine datasets from different soils/regions.
        Use for quantification (regression on MP concentration) or classification.
        """
        X_combined = np.vstack(X_list)
        y_combined = np.concatenate(y_list)
        
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
        
        self.svm_reg = SVR(kernel='rbf', C=100, gamma='scale')
        self.svm_reg.fit(X_train, y_train)
        
        y_pred = self.svm_reg.predict(X_test)
        print("=== Multisource LS-SVM / SVR Baseline ===")
        print(f"R²: {r2_score(y_test, y_pred):.4f}")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        return self.svm_reg


# Example integration
if __name__ == "__main__":
    # Assume X_spectra (n_samples, n_bands), y_labels or y_concentration from your data loader
    baselines = MPBaselineModels()
    
    # baselines.train_rf(X_spectra, y_labels)
    # baselines.train_1d_cnn(X_spectra, y_labels)
    # baselines.train_multisource_svm([X_soil1, X_soil2, ...], [y_soil1, y_soil2, ...])
    
    print(" Baseline models added – compare with your 3D CNN for robustness and quantification.")
