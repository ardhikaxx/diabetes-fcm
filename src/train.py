# D:\diabetes-fcm\src\train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import yaml
import warnings
warnings.filterwarnings('ignore')

class EnhancedModelTrainer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.config['training']['use_cuda'] 
            else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        self.early_stopping_patience = self.config['model'].get('early_stopping_patience', 20)
    
    def create_weighted_sampler(self, y):
        """Create weighted sampler for imbalanced data"""
        class_counts = np.bincount(y)
        class_weights = 1. / class_counts
        sample_weights = class_weights[y]
        
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(y),
            replacement=True
        )
        return sampler
    
    def prepare_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test,
                           fcm_membership_train=None, fcm_membership_val=None, fcm_membership_test=None):
        """Prepare PyTorch dataloaders with validation set"""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Move to device
        X_train_tensor = X_train_tensor.to(self.device)
        X_val_tensor = X_val_tensor.to(self.device)
        X_test_tensor = X_test_tensor.to(self.device)
        
        y_train_tensor = y_train_tensor.to(self.device)
        y_val_tensor = y_val_tensor.to(self.device)
        y_test_tensor = y_test_tensor.to(self.device)
        
        # Create weighted sampler for training
        train_sampler = self.create_weighted_sampler(y_train)
        
        # Create datasets
        if fcm_membership_train is not None:
            fcm_train_tensor = torch.FloatTensor(fcm_membership_train).to(self.device)
            fcm_val_tensor = torch.FloatTensor(fcm_membership_val).to(self.device)
            fcm_test_tensor = torch.FloatTensor(fcm_membership_test).to(self.device)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor, fcm_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor, fcm_val_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor, fcm_test_tensor)
        else:
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['model']['batch_size'],
            sampler=train_sampler,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['model']['batch_size'],
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['model']['batch_size'],
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, data in enumerate(progress_bar):
            if len(data) == 3:
                X_batch, y_batch, fcm_batch = data
            else:
                X_batch, y_batch = data
                fcm_batch = None
            
            optimizer.zero_grad()
            
            outputs = model(X_batch)
            
            if fcm_batch is not None and hasattr(criterion, 'alpha'):
                loss = criterion(outputs, y_batch, fcm_batch)
            else:
                loss = criterion(outputs, y_batch)
            
            # Add L2 regularization manually if not using weight_decay
            if self.config['model'].get('l2_reg', 0) > 0 and not self.config.get('regularization', {}).get('use_weight_decay', False):
                l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
                loss += self.config['model']['l2_reg'] * l2_reg
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
            
            progress_bar.set_postfix({
                'Loss': loss.item(),
                'Acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, model, data_loader, criterion):
        """Evaluate model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        all_probs = []
        all_memberships = []
        
        with torch.no_grad():
            for data in data_loader:
                if len(data) == 3:
                    X_batch, y_batch, fcm_batch = data
                else:
                    X_batch, y_batch = data
                    fcm_batch = None
                
                outputs = model(X_batch)
                
                if fcm_batch is not None and hasattr(criterion, 'alpha'):
                    loss = criterion(outputs, y_batch, fcm_batch)
                else:
                    loss = criterion(outputs, y_batch)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
                
                probs = torch.softmax(outputs, dim=1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_memberships.extend(outputs.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        # Calculate AUC if binary classification
        if len(np.unique(all_targets)) == 2:
            try:
                all_probs_np = np.array(all_probs)
                if all_probs_np.shape[1] >= 2:
                    auc = roc_auc_score(all_targets, all_probs_np[:, 1])
                else:
                    auc = None
            except:
                auc = None
        else:
            auc = None
        
        return avg_loss, accuracy, auc, np.array(all_preds), np.array(all_targets), np.array(all_memberships)
    
    def train(self, model, train_loader, val_loader, test_loader,
              criterion, optimizer, scheduler=None):
        """Complete training loop with early stopping"""
        
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        val_aucs = []
        
        best_val_acc = 0
        best_val_auc = 0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.config['model']['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )
            
            # Evaluate on validation set
            val_loss, val_acc, val_auc, _, _, _ = self.evaluate(
                model, val_loader, criterion
            )
            
            # Update scheduler
            if scheduler:
                scheduler.step(val_loss)
            
            # Store metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            if val_auc is not None:
                val_aucs.append(val_auc)
            
            # Early stopping and model saving logic
            # Use AUC if available, otherwise use accuracy
            if val_auc is not None:
                current_metric = val_auc
                best_metric = best_val_auc
            else:
                current_metric = val_acc
                best_metric = best_val_acc
            
            if current_metric > best_metric:
                if val_auc is not None:
                    best_val_auc = val_auc
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_auc': val_auc if val_auc is not None else 0,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f"{self.config['training']['save_path']}diabetes_classifier_best.pth")
                
                print(f"✓ Saved best model at epoch {epoch+1}")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
            
            print(f"Epoch {epoch+1}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                  f"{f', Val AUC: {val_auc:.4f}' if val_auc is not None else ''}")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\nLoaded best model with Val Acc: {best_val_acc:.2f}%")
            if val_aucs:
                print(f"Best Val AUC: {best_val_auc:.4f}")
        
        # Final evaluation on test set
        print("\n" + "="*50)
        print("Final Evaluation on Test Set")
        print("="*50)
        test_loss, test_acc, test_auc, y_pred, y_true, y_membership = self.evaluate(
            model, test_loader, criterion
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        if test_auc is not None:
            print(f"Test AUC: {test_auc:.4f}")
        
        # Save final model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_accuracy': test_acc,
            'test_auc': test_auc if test_auc is not None else 0,
        }, f"{self.config['training']['save_path']}diabetes_classifier_final.pth")
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'val_aucs': val_aucs,
            'test_metrics': {
                'loss': test_loss,
                'accuracy': test_acc,
                'auc': test_auc
            },
            'best_val_accuracy': best_val_acc,
            'best_val_auc': best_val_auc,
            'predictions': {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_membership': y_membership
            }
        }
    
    def plot_training_history(self, history, save_path=None):
        """Plot enhanced training history"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_losses'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_losses'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_accs'], label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(history['val_accs'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC plot (if available)
        if history['val_aucs']:
            axes[1, 0].plot(history['val_aucs'], label='Validation AUC', 
                          color='green', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC Score')
            axes[1, 0].set_title('Validation AUC Over Time')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].axis('off')
        
        # Test metrics summary
        axes[1, 1].axis('off')
        test_text = "Test Set Performance:\n\n"
        test_text += f"Accuracy: {history['test_metrics']['accuracy']:.2f}%\n"
        if history['test_metrics']['auc']:
            test_text += f"AUC: {history['test_metrics']['auc']:.4f}\n"
        test_text += f"Loss: {history['test_metrics']['loss']:.4f}\n\n"
        test_text += f"Best Validation:\n"
        test_text += f"Accuracy: {history['best_val_accuracy']:.2f}%\n"
        if history['best_val_auc']:
            test_text += f"AUC: {history['best_val_auc']:.4f}"
        
        axes[1, 1].text(0.1, 0.5, test_text, fontsize=12, 
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(self, y_true, y_probs, save_path=None):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve, auc
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
        except Exception as e:
            print(f"Could not plot ROC curve: {e}")
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Healthy', 'Diabetic'],
                       yticklabels=['Healthy', 'Diabetic'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix (Accuracy: {100*sum(y_true==y_pred)/len(y_true):.1f}%)')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
        except Exception as e:
            print(f"Could not plot confusion matrix: {e}")
    
    def generate_detailed_report(self, y_true, y_pred, y_probs=None, save_path=None):
        """Generate detailed classification report"""
        try:
            report = classification_report(y_true, y_pred, 
                                          target_names=['Healthy', 'Diabetic'])
            
            report += "\n" + "="*50 + "\n"
            report += "Model Performance Summary\n"
            report += "="*50 + "\n"
            report += f"Total Samples: {len(y_true)}\n"
            report += f"Accuracy: {100*sum(y_true==y_pred)/len(y_true):.2f}%\n"
            
            if y_probs is not None:
                try:
                    auc = roc_auc_score(y_true, y_probs[:, 1])
                    report += f"AUC Score: {auc:.4f}\n"
                except:
                    pass
            
            cm = confusion_matrix(y_true, y_pred)
            report += "\nConfusion Matrix:\n"
            report += f"True Negatives: {cm[0,0]}\n"
            report += f"False Positives: {cm[0,1]}\n"
            report += f"False Negatives: {cm[1,0]}\n"
            report += f"True Positives: {cm[1,1]}\n"
            
            # Calculate additional metrics
            precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
            recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            report += f"\nAdditional Metrics:\n"
            report += f"Precision: {precision:.4f}\n"
            report += f"Recall/Sensitivity: {recall:.4f}\n"
            report += f"F1-Score: {f1:.4f}\n"
            report += f"Specificity: {cm[0,0]/(cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0:.4f}\n"
            
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report)
            
            print(report)
            return report
        except Exception as e:
            error_msg = f"Error generating report: {e}"
            print(error_msg)
            return error_msg