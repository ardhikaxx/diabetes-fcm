# D:\diabetes-fcm\src\fcm_clustering.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class FuzzyCMeans:
    def __init__(self, n_clusters=4, m=2.0, max_iter=100, error=0.005):
        """
        Fuzzy C-Means Clustering
        
        Parameters:
        -----------
        n_clusters : int, number of clusters
        m : float, fuzziness parameter (m > 1)
        max_iter : int, maximum iterations
        error : float, stopping criterion
        """
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        
        self.centers = None
        self.membership = None
        self.history = []
        
    def initialize_membership(self, n_samples):
        """Randomly initialize membership matrix"""
        membership = np.random.rand(n_samples, self.n_clusters)
        membership = membership / np.sum(membership, axis=1, keepdims=True)
        return membership
    
    def calculate_centers(self, X, membership):
        """Calculate cluster centers"""
        numerator = np.dot((membership ** self.m).T, X)
        denominator = np.sum(membership ** self.m, axis=0, keepdims=True).T
        centers = numerator / denominator
        return centers
    
    def update_membership(self, X, centers):
        """Update membership matrix"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - centers[i], axis=1)
        
        # Handle zero distances
        distances = np.where(distances == 0, 1e-10, distances)
        
        power = 2 / (self.m - 1)
        inv_distances = distances ** (-power)
        sum_inv_distances = np.sum(inv_distances, axis=1, keepdims=True)
        
        membership = inv_distances / sum_inv_distances
        return membership
    
    def fit(self, X):
        """
        Fit FCM to data
        
        Returns:
        --------
        centers : cluster centers
        membership : fuzzy membership matrix
        """
        n_samples = X.shape[0]
        
        # Initialize
        membership = self.initialize_membership(n_samples)
        
        # Iterate
        for iteration in tqdm(range(self.max_iter), desc="FCM Training"):
            # Calculate centers
            centers = self.calculate_centers(X, membership)
            
            # Update membership
            new_membership = self.update_membership(X, centers)
            
            # Check convergence
            change = np.max(np.abs(new_membership - membership))
            self.history.append(change)
            
            if change < self.error:
                print(f"Converged at iteration {iteration+1}")
                break
                
            membership = new_membership
        
        self.centers = centers
        self.membership = membership
        
        return centers, membership
    
    def predict_cluster(self, X):
        """Predict cluster for new samples"""
        if self.centers is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        membership = self.update_membership(X, self.centers)
        return np.argmax(membership, axis=1), membership
    
    def calculate_severity_scores(self, membership, feature_weights=None):
        """
        Calculate diabetes severity scores based on membership
        
        Higher cluster index = more severe condition (adjust based on your data)
        """
        if membership is None or len(membership) == 0:
            return np.array([])
            
        if feature_weights is None:
            feature_weights = np.arange(1, self.n_clusters + 1)  # Cluster 1 = least severe, Cluster 4 = most severe
        
        # Weighted severity score
        severity_scores = np.dot(membership, feature_weights)
        
        # Normalize to 0-1 scale
        if severity_scores.max() > severity_scores.min():
            severity_scores = (severity_scores - severity_scores.min()) / \
                             (severity_scores.max() - severity_scores.min())
        else:
            severity_scores = np.zeros_like(severity_scores)
        
        return severity_scores
    
    def visualize_clusters(self, X, feature_names=None, save_path=None):
        """Visualize clusters using PCA"""
        try:
            from sklearn.decomposition import PCA
            
            # Check if we have enough features for PCA
            if X.shape[1] < 2:
                print("Not enough features for PCA visualization")
                return
            
            pca = PCA(n_components=min(2, X.shape[1]))
            X_pca = pca.fit_transform(X)
            
            # Get hard clusters
            if self.membership is not None:
                hard_clusters = np.argmax(self.membership, axis=1)
            else:
                print("No membership data available for visualization")
                return
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # PCA visualization
            scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                     c=hard_clusters, cmap='viridis', alpha=0.6)
            axes[0].set_title('FCM Clusters (PCA)')
            axes[0].set_xlabel('Principal Component 1')
            axes[0].set_ylabel('Principal Component 2')
            plt.colorbar(scatter, ax=axes[0])
            
            # Membership distribution
            for i in range(self.n_clusters):
                axes[1].hist(self.membership[:, i], alpha=0.5, 
                            bins=30, label=f'Cluster {i}')
            axes[1].set_title('Membership Distribution')
            axes[1].set_xlabel('Membership Degree')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            
            # Severity scores
            severity_scores = self.calculate_severity_scores(self.membership)
            axes[2].hist(severity_scores, bins=30, alpha=0.7, color='red')
            axes[2].set_title('Diabetes Severity Scores')
            axes[2].set_xlabel('Severity Score (0-1)')
            axes[2].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            print(f"Could not visualize clusters: {e}")
    
    def save_centers(self, path='models/fcm_centers.npy'):
        """Save cluster centers"""
        if self.centers is not None:
            np.save(path, self.centers)
        else:
            print("Warning: No centers to save.")
    
    def load_centers(self, path='models/fcm_centers.npy'):
        """Load cluster centers"""
        self.centers = np.load(path)
        return self.centers