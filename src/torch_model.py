# D:\diabetes-fcm\src\torch_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EnhancedDiabetesClassifier(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=4, dropout_rate=0.3):
        """
        Enhanced Neural Network for Diabetes Severity Classification
        with better regularization
        """
        super(EnhancedDiabetesClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
        )
        
        self.output_layer = nn.Linear(hidden_dim // 4, output_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_bn(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.output_layer(x)
    
    def predict_with_softmax(self, x):
        """Predict with softmax activation"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class RegularizedFuzzyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.1):
        """
        Enhanced loss function with multiple regularization terms
        """
        super(RegularizedFuzzyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
        
    def forward(self, predictions, targets, fcm_membership=None):
        """
        Calculate combined loss with regularization
        """
        # Cross entropy loss with label smoothing
        ce = self.ce_loss(predictions, targets)
        
        total_loss = ce
        
        # Add fuzzy consistency loss if available
        if fcm_membership is not None and self.alpha > 0:
            predictions_soft = F.softmax(predictions, dim=1)
            
            # Add small epsilon to avoid log(0)
            predictions_soft = torch.clamp(predictions_soft, min=1e-10)
            fcm_membership = torch.clamp(fcm_membership, min=1e-10)
            
            # KL divergence
            fuzzy_loss = F.kl_div(
                predictions_soft.log(), 
                fcm_membership, 
                reduction='batchmean'
            )
            
            total_loss = (1 - self.alpha) * ce + self.alpha * fuzzy_loss
        
        return total_loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, predictions, targets):
        ce_loss = self.ce_loss(predictions, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()