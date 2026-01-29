# D:\diabetes-fcm\main.py
import numpy as np
import pandas as pd
import yaml
import torch
import torch.optim as optim
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import DiabetesDataPreprocessor
from src.fcm_clustering import FuzzyCMeans
from src.torch_model import EnhancedDiabetesClassifier, RegularizedFuzzyLoss, FocalLoss
from src.train import EnhancedModelTrainer

def main():
    # Create directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("results/fcm_results").mkdir(parents=True, exist_ok=True)
    Path("results/model_performance").mkdir(parents=True, exist_ok=True)
    Path("results/severity_analysis").mkdir(parents=True, exist_ok=True)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("ENHANCED DIABETES SEVERITY CLASSIFICATION")
    print("With Anti-Overfitting Techniques")
    print("=" * 60)
    
    # Step 1: Data Preprocessing
    print("\n[STEP 1] Loading and Preprocessing Data...")
    preprocessor = DiabetesDataPreprocessor()
    df = preprocessor.load_data()
    X_scaled, y, feature_names = preprocessor.preprocess(df)
    
    print(f"\nProcessed Features: {len(feature_names)}")
    print(f"Features: {feature_names}")
    
    # Save processed data
    processed_df = pd.DataFrame(X_scaled, columns=feature_names)
    processed_df['Outcome'] = y
    processed_df.to_csv('data/processed/normalized_data.csv', index=False)
    
    # Split data into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X_scaled, y)
    preprocessor.save_scaler()
    
    # Step 2: Fuzzy C-Means Clustering
    print("\n[STEP 2] Performing Fuzzy C-Means Clustering...")
    fcm = FuzzyCMeans(
        n_clusters=config['fcm']['n_clusters'],
        m=config['fcm']['m'],
        max_iter=config['fcm']['max_iter'],
        error=config['fcm']['error']
    )
    
    centers, membership = fcm.fit(X_train)
    fcm.save_centers()
    
    # Visualize clusters
    try:
        fcm.visualize_clusters(
            X_train, 
            feature_names,
            save_path='results/fcm_results/cluster_visualization.png'
        )
    except Exception as e:
        print(f"Note: Could not visualize clusters: {e}")
    
    # Calculate severity scores for training data
    train_severity_scores = fcm.calculate_severity_scores(membership)
    
    # Save FCM results
    fcm_results = pd.DataFrame({
        'Patient_ID': range(len(X_train)),
        'Hard_Cluster': np.argmax(membership, axis=1),
        **{f'Membership_Cluster_{i}': membership[:, i] 
           for i in range(config['fcm']['n_clusters'])},
        'Severity_Score': train_severity_scores,
        'Actual_Outcome': y_train
    })
    fcm_results.to_csv('results/fcm_results/cluster_assignments.csv', index=False)
    
    # Step 3: Prepare FCM membership for validation and test data
    print("\n[STEP 3] Predicting FCM Membership for Validation and Test Data...")
    val_clusters, val_membership = fcm.predict_cluster(X_val)
    test_clusters, test_membership = fcm.predict_cluster(X_test)
    
    val_severity = fcm.calculate_severity_scores(val_membership)
    test_severity = fcm.calculate_severity_scores(test_membership)
    
    # Step 4: Prepare Neural Network with enhanced training
    print("\n[STEP 4] Training Enhanced Neural Network...")
    trainer = EnhancedModelTrainer()
    
    # Create dataloaders with FCM membership
    train_loader, val_loader, test_loader = trainer.prepare_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        fcm_membership_train=membership,
        fcm_membership_val=val_membership,
        fcm_membership_test=test_membership
    )
    
    # Initialize enhanced model
    input_dim = len(feature_names)  # Updated with new features
    model = EnhancedDiabetesClassifier(
        input_dim=input_dim,
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['fcm']['n_clusters'],
        dropout_rate=config.get('regularization', {}).get('dropout_rate', 0.3)
    ).to(trainer.device)
    
    print(f"\nModel Architecture:")
    print(f"Input features: {input_dim}")
    print(f"Hidden dimensions: {config['model']['hidden_dim']}")
    print(f"Output clusters: {config['fcm']['n_clusters']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Define loss and optimizer with regularization
    criterion = RegularizedFuzzyLoss(alpha=0.3)
    # criterion = FocalLoss()  # Alternative for class imbalance
    
    # Configure optimizer with weight decay
    weight_decay = config['model'].get('l2_reg', 0.001) if config.get('regularization', {}).get('use_weight_decay', True) else 0
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['model']['learning_rate'],
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['model']['epochs'],
        eta_min=config['model']['learning_rate'] * 0.01
    )
    
    # Train model with early stopping
    print("\n[STEP 5] Starting Training with Early Stopping...")
    history = trainer.train(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler
    )
    
    # Get predictions for detailed analysis
    _, _, _, y_pred, y_true, y_membership = trainer.evaluate(
        model, test_loader, criterion
    )
    
    # Get probability predictions for ROC curve
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(trainer.device)
        test_logits = model(X_test_tensor)
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
    
    # Step 5: Generate comprehensive reports
    print("\n[STEP 6] Generating Performance Reports...")
    
    # Plot training history
    trainer.plot_training_history(
        history,
        save_path='results/model_performance/training_history.png'
    )
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(
        y_true, y_pred,
        save_path='results/model_performance/confusion_matrix.png'
    )
    
    # Plot ROC curve
    if test_probs.shape[1] >= 2:
        trainer.plot_roc_curve(
            y_true, test_probs[:, 1],
            save_path='results/model_performance/roc_curve.png'
        )
    
    # Generate detailed report
    report = trainer.generate_detailed_report(
        y_true, y_pred, test_probs,
        save_path='results/model_performance/detailed_metrics_report.txt'
    )
    
    # Step 6: Severity Analysis
    print("\n[STEP 7] Performing Comprehensive Severity Analysis...")
    
    # Get severity predictions for all patients
    with torch.no_grad():
        X_all_tensor = torch.FloatTensor(X_scaled).to(trainer.device)
        
        # Get FCM centers
        centers_to_use = fcm.centers if hasattr(fcm, 'centers') and fcm.centers is not None else None
        
        # Predict with model
        all_logits = model(X_all_tensor)
        all_membership_nn = torch.softmax(all_logits, dim=1)
        
        # Calculate severity scores
        if centers_to_use is not None:
            severity_weights = torch.arange(1, config['fcm']['n_clusters'] + 1, 
                                           dtype=torch.float32).to(trainer.device)
            all_severity_nn = torch.matmul(all_membership_nn, severity_weights)
            all_severity_nn = (all_severity_nn - all_severity_nn.min()) / \
                             (all_severity_nn.max() - all_severity_nn.min() + 1e-8)
        else:
            if all_membership_nn.shape[1] >= 2:
                all_severity_nn = all_membership_nn[:, 1]
            else:
                all_severity_nn = all_membership_nn[:, 0]
    
    # Convert tensors to numpy
    all_membership_nn_np = all_membership_nn.cpu().numpy()
    all_severity_nn_np = all_severity_nn.cpu().numpy()
    
    # Combine all data
    total_patients = len(X_scaled)
    
    # Combine FCM membership for all data
    fcm_membership_all = np.vstack([membership, val_membership, test_membership])
    fcm_severity_all = np.concatenate([train_severity_scores, val_severity, test_severity])
    
    # Get original features for all patients
    # Since we added new features, we need to get the original data
    original_features = df[['Glucose', 'BMI', 'Age']].values
    
    # Create comprehensive patient profiles
    patient_profiles = pd.DataFrame({
        'Patient_ID': range(total_patients),
        'Glucose': original_features[:, 0],
        'BMI': original_features[:, 1],
        'Age': original_features[:, 2],
        **{f'FCM_Membership_{i}': fcm_membership_all[:, i] 
           for i in range(config['fcm']['n_clusters'])},
        'FCM_Severity_Score': fcm_severity_all,
        **{f'NN_Membership_{i}': all_membership_nn_np[:, i]
           for i in range(config['fcm']['n_clusters'])},
        'NN_Severity_Score': all_severity_nn_np,
        'Predicted_Cluster_NN': np.argmax(all_membership_nn_np, axis=1),
        'Predicted_Cluster_FCM': np.argmax(fcm_membership_all, axis=1),
        'Actual_Outcome': y,
        'NN_Risk_Category': pd.cut(
            all_severity_nn_np,
            bins=[0, 0.25, 0.5, 0.75, 1],
            labels=['Low', 'Medium', 'High', 'Critical']
        ),
        'FCM_Risk_Category': pd.cut(
            fcm_severity_all,
            bins=[0, 0.25, 0.5, 0.75, 1],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
    })
    
    patient_profiles.to_csv(
        'results/severity_analysis/comprehensive_patient_profiles.csv', 
        index=False
    )
    
    # Create summary statistics
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Patients',
            'Diabetic Patients',
            'Healthy Patients',
            'FCM Clusters',
            'Best Validation Accuracy',
            'Test Accuracy',
            'Test AUC',
            'Model Parameters'
        ],
        'Value': [
            len(df),
            f"{sum(y)} ({sum(y)/len(y)*100:.1f}%)",
            f"{len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)",
            config['fcm']['n_clusters'],
            f"{history.get('best_val_accuracy', 0):.2f}%",
            f"{history.get('test_metrics', {}).get('accuracy', 0):.2f}%",
            f"{history.get('test_metrics', {}).get('auc', 0):.4f}" if history.get('test_metrics', {}).get('auc') else 'N/A',
            f"{sum(p.numel() for p in model.parameters()):,}"
        ]
    })
    
    summary_stats.to_csv('results/model_performance/summary_statistics.csv', index=False)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("PROJECT COMPLETION SUMMARY")
    print("=" * 60)
    print(f"✅ Total Patients: {len(df)}")
    print(f"✅ Diabetic Patients: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"✅ Healthy Patients: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    print(f"✅ FCM Clusters: {config['fcm']['n_clusters']}")
    print(f"✅ Best Validation Accuracy: {history.get('best_val_accuracy', 0):.2f}%")
    print(f"✅ Test Accuracy: {history.get('test_metrics', {}).get('accuracy', 0):.2f}%")
    if history.get('test_metrics', {}).get('auc'):
        print(f"✅ Test AUC: {history.get('test_metrics', {}).get('auc'):.4f}")
    print(f"✅ Model Saved: models/diabetes_classifier_best.pth")
    print(f"✅ Patient Profiles Saved: results/severity_analysis/comprehensive_patient_profiles.csv")
    print("=" * 60)
    
    # Example prediction with confidence
    print("\n[EXAMPLE] Predicting for sample patients with confidence...")
    sample_indices = [0, 50, 100, 200]
    
    for idx in sample_indices:
        if idx < len(X_scaled):
            example_patient = X_scaled[idx:idx+1]
            example_tensor = torch.FloatTensor(example_patient).to(trainer.device)
            
            with torch.no_grad():
                logits = model(example_tensor)
                membership_pred = torch.softmax(logits, dim=1)
                
                # Predict severity
                if centers_to_use is not None:
                    severity_weights = torch.arange(1, config['fcm']['n_clusters'] + 1, 
                                                   dtype=torch.float32).to(trainer.device)
                    severity_pred = torch.matmul(membership_pred, severity_weights)
                    severity_pred = (severity_pred - severity_pred.min()) / \
                                   (severity_pred.max() - severity_pred.min() + 1e-8)
                else:
                    if membership_pred.shape[1] >= 2:
                        severity_pred = membership_pred[:, 1]
                    else:
                        severity_pred = membership_pred[:, 0]
            
            pred_cluster = torch.argmax(membership_pred).item()
            confidence = torch.max(membership_pred).item()
            
            print(f"\nPatient {idx}:")
            print(f"  Features: Glucose={df.iloc[idx]['Glucose']}, "
                  f"BMI={df.iloc[idx]['BMI']:.1f}, Age={df.iloc[idx]['Age']}")
            print(f"  Predicted Cluster: {pred_cluster} (Confidence: {confidence:.2%})")
            print(f"  Severity Score: {severity_pred[0].cpu().numpy():.3f}")
            print(f"  Risk Category: {patient_profiles.iloc[idx]['NN_Risk_Category']}")
            print(f"  Actual Outcome: {'Diabetic' if y[idx] == 1 else 'Healthy'}")
    
    print("\n" + "=" * 60)
    print("✅ Project completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()