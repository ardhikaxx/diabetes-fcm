# D:\diabetes-fcm\src\monitoring.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
import pandas as pd

class OverfittingMonitor:
    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'gap_loss': [],
            'gap_acc': []
        }
    
    def update(self, train_loss, val_loss, train_acc, val_acc):
        """Update metrics history"""
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['train_acc'].append(train_acc)
        self.metrics_history['val_acc'].append(val_acc)
        
        # Calculate gaps
        loss_gap = train_loss - val_loss
        acc_gap = train_acc - val_acc
        self.metrics_history['gap_loss'].append(loss_gap)
        self.metrics_history['gap_acc'].append(acc_gap)
    
    def plot_overfitting_diagnostics(self, save_path=None):
        """Plot comprehensive overfitting diagnostics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.metrics_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.metrics_history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.metrics_history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss gap
        axes[0, 2].plot(epochs, self.metrics_history['gap_loss'], 'g-', linewidth=2)
        axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 2].fill_between(epochs, 0, self.metrics_history['gap_loss'], 
                               where=np.array(self.metrics_history['gap_loss']) > 0,
                               color='red', alpha=0.3, label='Overfitting')
        axes[0, 2].fill_between(epochs, 0, self.metrics_history['gap_loss'],
                               where=np.array(self.metrics_history['gap_loss']) < 0,
                               color='green', alpha=0.3, label='Underfitting')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Train Loss - Val Loss')
        axes[0, 2].set_title('Loss Gap (Overfitting Indicator)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Accuracy gap
        axes[1, 0].plot(epochs, self.metrics_history['gap_acc'], 'g-', linewidth=2)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].fill_between(epochs, 0, self.metrics_history['gap_acc'],
                               where=np.array(self.metrics_history['gap_acc']) > 0,
                               color='red', alpha=0.3, label='Overfitting')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Train Acc - Val Acc')
        axes[1, 0].set_title('Accuracy Gap')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Final metrics comparison
        metrics = ['Final Train Loss', 'Final Val Loss', 'Final Train Acc', 'Final Val Acc']
        values = [
            self.metrics_history['train_loss'][-1],
            self.metrics_history['val_loss'][-1],
            self.metrics_history['train_acc'][-1],
            self.metrics_history['val_acc'][-1]
        ]
        
        axes[1, 1].bar(metrics, values, color=['blue', 'red', 'blue', 'red'])
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Final Metrics Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Summary statistics
        axes[1, 2].axis('off')
        summary_text = "Overfitting Analysis Summary:\n\n"
        final_loss_gap = self.metrics_history['gap_loss'][-1]
        final_acc_gap = self.metrics_history['gap_acc'][-1]
        
        if final_loss_gap > 0.1 and final_acc_gap > 5:
            summary_text += "Status: ⚠️ Potential Overfitting\n"
        elif final_loss_gap < -0.1 and final_acc_gap < -5:
            summary_text += "Status: ⚠️ Potential Underfitting\n"
        else:
            summary_text += "Status: ✅ Good Fit\n"
        
        summary_text += f"\nFinal Loss Gap: {final_loss_gap:.4f}\n"
        summary_text += f"Final Accuracy Gap: {final_acc_gap:.2f}%\n"
        summary_text += f"Max Loss Gap: {max(self.metrics_history['gap_loss']):.4f}\n"
        summary_text += f"Max Accuracy Gap: {max(self.metrics_history['gap_acc']):.2f}%\n"
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, 
                       verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_overfitting_report(self, save_path=None):
        """Generate detailed overfitting report"""
        report = "=" * 60 + "\n"
        report += "OVERFITTING ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        final_loss_gap = self.metrics_history['gap_loss'][-1]
        final_acc_gap = self.metrics_history['gap_acc'][-1]
        
        report += "Diagnosis:\n"
        if final_loss_gap > 0.1 and final_acc_gap > 5:
            report += "⚠️  POTENTIAL OVERFITTING DETECTED\n"
            report += "   - Training loss significantly lower than validation loss\n"
            report += "   - Training accuracy significantly higher than validation accuracy\n"
            report += "\nRecommendations:\n"
            report += "1. Increase regularization (dropout, weight decay)\n"
            report += "2. Add more training data\n"
            report += "3. Simplify model architecture\n"
            report += "4. Use data augmentation\n"
            report += "5. Implement early stopping\n"
        elif final_loss_gap < -0.1 and final_acc_gap < -5:
            report += "⚠️  POTENTIAL UNDERFITTING DETECTED\n"
            report += "   - Training loss higher than validation loss\n"
            report += "   - Training accuracy lower than validation accuracy\n"
            report += "\nRecommendations:\n"
            report += "1. Increase model complexity\n"
            report += "2. Train for more epochs\n"
            report += "3. Reduce regularization\n"
            report += "4. Increase learning rate\n"
            report += "5. Use more sophisticated architecture\n"
        else:
            report += "✅  GOOD FIT ACHIEVED\n"
            report += "   - Training and validation metrics are well-balanced\n"
            report += "   - Model generalizes well to unseen data\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += "QUANTITATIVE ANALYSIS\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Final Training Loss: {self.metrics_history['train_loss'][-1]:.4f}\n"
        report += f"Final Validation Loss: {self.metrics_history['val_loss'][-1]:.4f}\n"
        report += f"Loss Gap: {final_loss_gap:.4f}\n\n"
        
        report += f"Final Training Accuracy: {self.metrics_history['train_acc'][-1]:.2f}%\n"
        report += f"Final Validation Accuracy: {self.metrics_history['val_acc'][-1]:.2f}%\n"
        report += f"Accuracy Gap: {final_acc_gap:.2f}%\n\n"
        
        report += f"Maximum Loss Gap: {max(self.metrics_history['gap_loss']):.4f}\n"
        report += f"Maximum Accuracy Gap: {max(self.metrics_history['gap_acc']):.2f}%\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        print(report)
        return report

def plot_calibration_curve(y_true, y_probs, save_path=None):
    """Plot calibration curve to check model calibration"""
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 's-', label='Model')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()