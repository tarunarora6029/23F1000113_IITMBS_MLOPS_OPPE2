#!/usr/bin/env python3
"""
Step 7: Security Testing - Data Poisoning Attack
Simulate label interchange attack and compare performance
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create results directory"""
    os.makedirs('results/step7_security', exist_ok=True)

def load_original_data():
    """Load and preprocess original data"""
    print("📊 Loading original data...")
    
    # Load data
    df = pd.read_csv('data/data.csv')
    
    # Load metadata for consistent preprocessing
    with open('models/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Apply same preprocessing
    gender_mapping = metadata['gender_mapping']
    df['gender'] = df['gender'].map(gender_mapping)
    df_clean = df.dropna()
    
    # Features and target
    feature_names = metadata['feature_names']
    X = df_clean[feature_names]
    y = df_clean['target']
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Original data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y_encoded)}")
    
    return X, y_encoded, feature_names, le

def create_poisoned_datasets(X, y, poison_ratios=[0.05, 0.1, 0.15, 0.2, 0.25]):
    """Create datasets with different levels of label poisoning"""
    print("☠️ Creating poisoned datasets...")
    
    poisoned_datasets = {}
    
    for ratio in poison_ratios:
        print(f"Creating dataset with {ratio*100:.0f}% label poisoning...")
        
        # Copy original data
        X_poison = X.copy()
        y_poison = y.copy()
        
        # Randomly select samples to poison
        n_poison = int(len(y) * ratio)
        poison_indices = np.random.choice(len(y), n_poison, replace=False)
        
        # Flip labels (simple label interchange attack)
        y_poison[poison_indices] = 1 - y_poison[poison_indices]
        
        poisoned_datasets[ratio] = {
            'X': X_poison,
            'y': y_poison,
            'poison_indices': poison_indices,
            'n_poisoned': n_poison
        }
        
        print(f"  Poisoned {n_poison} samples out of {len(y)}")
        print(f"  New class distribution: {np.bincount(y_poison)}")
    
    return poisoned_datasets

def train_and_evaluate_models(X, y, poisoned_datasets, feature_names):
    """Train models on clean and poisoned data"""
    print("🔧 Training models on clean and poisoned data...")
    
    results = {}
    
    # Split data
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train clean model (baseline)
    print("Training clean model...")
    clean_model = LogisticRegression(random_state=42, max_iter=1000)
    clean_model.fit(X_train_scaled, y_train)
    
    # Evaluate clean model
    y_pred_clean = clean_model.predict(X_test_scaled)
    clean_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_clean),
        'precision': precision_score(y_test, y_pred_clean),
        'recall': recall_score(y_test, y_pred_clean),
        'f1': f1_score(y_test, y_pred_clean)
    }
    
    results['clean'] = {
        'model': clean_model,
        'metrics': clean_metrics,
        'predictions': y_pred_clean,
        'poison_ratio': 0.0
    }
    
    print(f"Clean model accuracy: {clean_metrics['accuracy']:.4f}")
    
    # Train poisoned models
    for poison_ratio, poison_data in poisoned_datasets.items():
        print(f"Training model with {poison_ratio*100:.0f}% poisoning...")
        
        # Use poisoned training data
        X_poison_train, X_poison_test, y_poison_train, y_poison_test = train_test_split(
            poison_data['X'], poison_data['y'], test_size=0.2, random_state=42, stratify=poison_data['y']
        )
        
        # Scale features
        scaler_poison = StandardScaler()
        X_poison_train_scaled = scaler_poison.fit_transform(X_poison_train)
        X_poison_test_scaled = scaler_poison.transform(X_poison_test)
        
        # Train poisoned model
        poison_model = LogisticRegression(random_state=42, max_iter=1000)
        poison_model.fit(X_poison_train_scaled, y_poison_train)
        
        # Evaluate on clean test set (important!)
        y_pred_poison = poison_model.predict(X_test_scaled)
        poison_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_poison),
            'precision': precision_score(y_test, y_pred_poison),
            'recall': recall_score(y_test, y_pred_poison),
            'f1': f1_score(y_test, y_pred_poison)
        }
        
        results[f'poison_{poison_ratio}'] = {
            'model': poison_model,
            'metrics': poison_metrics,
            'predictions': y_pred_poison,
            'poison_ratio': poison_ratio,
            'scaler': scaler_poison
        }
        
        print(f"Poisoned model ({poison_ratio*100:.0f}%) accuracy: {poison_metrics['accuracy']:.4f}")
    
    return results, X_test, y_test, X_test_scaled

def analyze_attack_impact(results):
    """Analyze the impact of poisoning attacks"""
    print("📊 Analyzing attack impact...")
    
    # Extract metrics for comparison
    poison_ratios = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    # Sort results by poison ratio
    sorted_results = sorted(results.items(), key=lambda x: x[1]['poison_ratio'])
    
    for model_name, result in sorted_results:
        poison_ratios.append(result['poison_ratio'] * 100)  # Convert to percentage
        accuracies.append(result['metrics']['accuracy'])
        precisions.append(result['metrics']['precision'])
        recalls.append(result['metrics']['recall'])
        f1_scores.append(result['metrics']['f1'])
    
    impact_analysis = {
        'poison_ratios': poison_ratios,
        'accuracies': accuracies,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores
    }
    
    # Calculate performance degradation
    baseline_accuracy = accuracies[0]  # Clean model accuracy
    max_degradation = baseline_accuracy - min(accuracies)
    
    impact_analysis['baseline_accuracy'] = baseline_accuracy
    impact_analysis['max_accuracy_loss'] = max_degradation
    impact_analysis['max_degradation_percent'] = (max_degradation / baseline_accuracy) * 100
    
    return impact_analysis

def visualize_attack_results(impact_analysis, results):
    """Create visualizations for attack results"""
    print("📈 Creating attack visualization...")
    
    # Performance degradation plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    poison_ratios = impact_analysis['poison_ratios']
    
    # Accuracy degradation
    axes[0,0].plot(poison_ratios, impact_analysis['accuracies'], 'b-o', linewidth=2, markersize=6)
    axes[0,0].set_title('Accuracy vs Poisoning Level', fontweight='bold')
    axes[0,0].set_xlabel('Poisoning Ratio (%)')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim([0.5, 1.0])
    
    # Precision degradation
    axes[0,1].plot(poison_ratios, impact_analysis['precisions'], 'g-s', linewidth=2, markersize=6)
    axes[0,1].set_title('Precision vs Poisoning Level', fontweight='bold')
    axes[0,1].set_xlabel('Poisoning Ratio (%)')
    axes[0,1].set_ylabel('Precision')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim([0.5, 1.0])
    
    # Recall degradation
    axes[1,0].plot(poison_ratios, impact_analysis['recalls'], 'r-^', linewidth=2, markersize=6)
    axes[1,0].set_title('Recall vs Poisoning Level', fontweight='bold')
    axes[1,0].set_xlabel('Poisoning Ratio (%)')
    axes[1,0].set_ylabel('Recall')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylim([0.5, 1.0])
    
    # F1 Score degradation
    axes[1,1].plot(poison_ratios, impact_analysis['f1_scores'], 'm-d', linewidth=2, markersize=6)
    axes[1,1].set_title('F1-Score vs Poisoning Level', fontweight='bold')
    axes[1,1].set_xlabel('Poisoning Ratio (%)')
    axes[1,1].set_ylabel('F1-Score')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig('results/step7_security/performance_degradation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Combined metrics plot
    plt.figure(figsize=(12, 8))
    
    plt.plot(poison_ratios, impact_analysis['accuracies'], 'b-o', label='Accuracy', linewidth=2, markersize=6)
    plt.plot(poison_ratios, impact_analysis['precisions'], 'g-s', label='Precision', linewidth=2, markersize=6)
    plt.plot(poison_ratios, impact_analysis['recalls'], 'r-^', label='Recall', linewidth=2, markersize=6)
    plt.plot(poison_ratios, impact_analysis['f1_scores'], 'm-d', label='F1-Score', linewidth=2, markersize=6)
    
    plt.title('Model Performance Under Data Poisoning Attack', fontsize=14, fontweight='bold')
    plt.xlabel('Poisoning Ratio (%)', fontsize=12)
    plt.ylabel('Performance Metric', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.5, 1.0])
    
    # Add annotations for critical points
    baseline_acc = impact_analysis['baseline_accuracy']
    worst_acc = min(impact_analysis['accuracies'])
    worst_idx = impact_analysis['accuracies'].index(worst_acc)
    worst_ratio = poison_ratios[worst_idx]
    
    plt.annotate(f'Baseline: {baseline_acc:.3f}', 
                xy=(0, baseline_acc), xytext=(5, baseline_acc+0.05),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                fontsize=10, color='blue')
    
    plt.annotate(f'Worst: {worst_acc:.3f}', 
                xy=(worst_ratio, worst_acc), xytext=(worst_ratio+5, worst_acc-0.05),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('results/step7_security/combined_performance.png', dpi=150, bbox_inches='tight')
    plt.close()

def feature_importance_comparison(results, feature_names):
    """Compare feature importance between clean and poisoned models"""
    print("🔍 Comparing feature importance...")
    
    # Get clean model coefficients
    clean_model = results['clean']['model']
    clean_coeff = clean_model.coef_[0]
    
    # Get most poisoned model coefficients
    max_poison_key = max([k for k in results.keys() if k.startswith('poison_')], 
                        key=lambda x: results[x]['poison_ratio'])
    poison_model = results[max_poison_key]['model']
    poison_coeff = poison_model.coef_[0]
    
    # Create comparison DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'clean_importance': np.abs(clean_coeff),
        'poison_importance': np.abs(poison_coeff),
        'clean_coeff': clean_coeff,
        'poison_coeff': poison_coeff
    })
    
    # Calculate difference
    importance_df['importance_diff'] = importance_df['poison_importance'] - importance_df['clean_importance']
    importance_df['coeff_diff'] = importance_df['poison_coeff'] - importance_df['clean_coeff']
    
    # Sort by absolute difference
    importance_df = importance_df.reindex(importance_df['importance_diff'].abs().sort_values(ascending=False).index)
    
    # Plot feature importance comparison
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    plt.bar(x - width/2, importance_df['clean_importance'], width, 
            label='Clean Model', alpha=0.8, color='blue')
    plt.bar(x + width/2, importance_df['poison_importance'], width, 
            label=f'Poisoned Model ({results[max_poison_key]["poison_ratio"]*100:.0f}%)', alpha=0.8, color='red')
    
    plt.title('Feature Importance: Clean vs Poisoned Model', fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Absolute Coefficient Value', fontsize=12)
    plt.xticks(x, importance_df['feature'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/step7_security/feature_importance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return importance_df

def detect_poisoned_samples(results, X_test, y_test):
    """Attempt to detect potentially poisoned samples using model disagreement"""
    print("🕵️ Attempting to detect poisoned samples...")
    
    # Get predictions from clean and most poisoned model
    clean_pred = results['clean']['predictions']
    
    max_poison_key = max([k for k in results.keys() if k.startswith('poison_')], 
                        key=lambda x: results[x]['poison_ratio'])
    poison_pred = results[max_poison_key]['predictions']
    
    # Find disagreements
    disagreements = (clean_pred != poison_pred)
    n_disagreements = np.sum(disagreements)
    
    # Get prediction probabilities
    clean_model = results['clean']['model']
    poison_model = results[max_poison_key]['model']
    
    # We need the scaled test data - use the same scaler as clean model
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_dummy = X_test  # This is just for fitting, we'll use stored scaler from original training
    
    # Load the original scaler
    original_scaler = joblib.load('models/scaler.pkl')
    X_test_scaled = original_scaler.transform(X_test)
    
    clean_proba = clean_model.predict_proba(X_test_scaled)
    poison_proba = poison_model.predict_proba(X_test_scaled)
    
    # Calculate confidence differences
    clean_confidence = np.max(clean_proba, axis=1)
    poison_confidence = np.max(poison_proba, axis=1)
    confidence_diff = np.abs(clean_confidence - poison_confidence)
    
    detection_results = {
        'total_test_samples': len(X_test),
        'disagreements': n_disagreements,
        'disagreement_rate': n_disagreements / len(X_test),
        'avg_confidence_diff': np.mean(confidence_diff),
        'max_confidence_diff': np.max(confidence_diff),
        'suspicious_samples': np.sum(confidence_diff > 0.5)  # Threshold for suspicion
    }
    
    return detection_results

def generate_security_report(impact_analysis, importance_df, detection_results):
    """Generate comprehensive security analysis report"""
    
    report = []
    report.append("🛡️ SECURITY ANALYSIS REPORT - DATA POISONING ATTACK")
    report.append("="*60)
    report.append(f"Analysis Time: {pd.Timestamp.now()}")
    
    # Attack summary
    report.append(f"\n🎯 ATTACK SUMMARY:")
    report.append(f"Attack Type: Label Interchange (Label Flipping)")
    report.append(f"Poisoning Levels Tested: {', '.join([f'{r:.0f}%' for r in impact_analysis['poison_ratios'][1:]])}")
    report.append(f"Baseline Model Accuracy: {impact_analysis['baseline_accuracy']:.4f}")
    report.append(f"Maximum Accuracy Loss: {impact_analysis['max_accuracy_loss']:.4f}")
    report.append(f"Maximum Performance Degradation: {impact_analysis['max_degradation_percent']:.1f}%")
    
    # Vulnerability assessment
    report.append(f"\n🚨 VULNERABILITY ASSESSMENT:")
    
    max_degradation = impact_analysis['max_degradation_percent']
    if max_degradation > 30:
        risk_level = "CRITICAL"
        risk_color = "🔴"
    elif max_degradation > 20:
        risk_level = "HIGH"
        risk_color = "🟠"
    elif max_degradation > 10:
        risk_level = "MEDIUM"
        risk_color = "🟡"
    else:
        risk_level = "LOW"
        risk_color = "🟢"
    
    report.append(f"Risk Level: {risk_color} {risk_level}")
    report.append(f"Model shows {max_degradation:.1f}% performance degradation under attack")
    
    # Performance impact by metric
    report.append(f"\n📊 PERFORMANCE IMPACT BY METRIC:")
    baseline_acc = impact_analysis['accuracies'][0]
    baseline_prec = impact_analysis['precisions'][0]
    baseline_rec = impact_analysis['recalls'][0]
    baseline_f1 = impact_analysis['f1_scores'][0]
    
    worst_acc = min(impact_analysis['accuracies'])
    worst_prec = min(impact_analysis['precisions'])
    worst_rec = min(impact_analysis['recalls'])
    worst_f1 = min(impact_analysis['f1_scores'])
    
    report.append(f"Accuracy:  {baseline_acc:.4f} → {worst_acc:.4f} (Δ {worst_acc-baseline_acc:+.4f})")
    report.append(f"Precision: {baseline_prec:.4f} → {worst_prec:.4f} (Δ {worst_prec-baseline_prec:+.4f})")
    report.append(f"Recall:    {baseline_rec:.4f} → {worst_rec:.4f} (Δ {worst_rec-baseline_rec:+.4f})")
    report.append(f"F1-Score:  {baseline_f1:.4f} → {worst_f1:.4f} (Δ {worst_f1-baseline_f1:+.4f})")
    
    # Feature importance changes
    report.append(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")
    top_affected = importance_df.head(3)
    report.append(f"Top 3 features most affected by poisoning:")
    for i, (_, row) in enumerate(top_affected.iterrows(), 1):
        report.append(f"{i}. {row['feature']}: Δ {row['importance_diff']:+.4f}")
    
    # Detection capabilities
    report.append(f"\n🕵️ POISON DETECTION ANALYSIS:")
    report.append(f"Model Disagreement Rate: {detection_results['disagreement_rate']:.1%}")
    report.append(f"Average Confidence Difference: {detection_results['avg_confidence_diff']:.4f}")
    report.append(f"Suspicious Samples Detected: {detection_results['suspicious_samples']}")
    
    if detection_results['disagreement_rate'] > 0.1:
        report.append("⚠️ High disagreement rate suggests potential poisoning")
    
    # Recommendations
    report.append(f"\n💡 SECURITY RECOMMENDATIONS:")
    
    if max_degradation > 20:
        report.append("🚨 IMMEDIATE ACTIONS REQUIRED:")
        report.append("1. Implement robust training data validation")
        report.append("2. Deploy anomaly detection for training samples")
        report.append("3. Use ensemble methods to increase robustness")
        report.append("4. Implement differential privacy techniques")
    
    report.append("\n🛡️ GENERAL SECURITY MEASURES:")
    report.append("1. Establish secure data collection pipelines")
    report.append("2. Implement data provenance tracking")
    report.append("3. Use statistical tests to detect label anomalies")
    report.append("4. Deploy multiple models for cross-validation")
    report.append("5. Regular model retraining with verified data")
    report.append("6. Implement model monitoring and drift detection")
    
    # Mitigation strategies
    report.append(f"\n🔧 MITIGATION STRATEGIES:")
    report.append("1. Data Sanitization: Remove suspicious samples before training")
    report.append("2. Robust Training: Use techniques like RONI (Reject on Negative Impact)")
    report.append("3. Byzantine-Robust Algorithms: Use aggregation methods resilient to outliers")
    report.append("4. Certified Defense: Implement provably robust training methods")
    report.append("5. Federated Learning: Reduce single points of failure")
    
    report.append(f"\n📁 Generated Files:")
    report.append("- performance_degradation.png (Individual metrics)")
    report.append("- combined_performance.png (All metrics together)")
    report.append("- feature_importance_comparison.png (Feature analysis)")
    report.append("- security_results.json (Detailed metrics)")
    
    return "\n".join(report)

def save_results(results, impact_analysis, importance_df, detection_results, report):
    """Save all security testing results"""
    print("💾 Saving security analysis results...")
    
    # Prepare results for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        serializable_results[key] = {
            'metrics': value['metrics'],
            'poison_ratio': value['poison_ratio']
            # Exclude model objects and arrays for JSON compatibility
        }
    
    final_results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'attack_type': 'Label Interchange',
        'model_results': serializable_results,
        'impact_analysis': impact_analysis,
        'feature_importance_changes': importance_df.to_dict('records'),
        'detection_analysis': detection_results,
        'security_report': report
    }
    
    with open('results/step7_security/security_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Save feature importance comparison
    importance_df.to_csv('results/step7_security/feature_importance_comparison.csv', index=False)
    
    # Save report as text
    with open('results/step7_security/security_report.txt', 'w') as f:
        f.write(report)

def main():
    """Main execution function"""
    print("🚀 Starting Step 7: Security Testing - Data Poisoning Attack")
    print("="*60)
    
    create_directories()
    
    # Load original data
    X, y, feature_names, label_encoder = load_original_data()
    
    # Create poisoned datasets
    poisoned_datasets = create_poisoned_datasets(X, y)
    
    # Train and evaluate models
    results, X_test, y_test, X_test_scaled = train_and_evaluate_models(X, y, poisoned_datasets, feature_names)
    
    # Analyze attack impact
    impact_analysis = analyze_attack_impact(results)
    
    # Create visualizations
    visualize_attack_results(impact_analysis, results)
    
    # Feature importance analysis
    importance_df = feature_importance_comparison(results, feature_names)
    
    # Detection analysis
    detection_results = detect_poisoned_samples(results, X_test, y_test)
    
    # Generate comprehensive report
    report = generate_security_report(impact_analysis, importance_df, detection_results)
    print(report)
    
    # Save all results
    save_results(results, impact_analysis, importance_df, detection_results, report)
    
    print("\n✅ Step 7: Security testing completed!")
    print("🛡️ Results saved to: results/step7_security/")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
