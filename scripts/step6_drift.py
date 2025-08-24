#!/usr/bin/env python3
"""
Step 6: Input Drift Detection
Compare training data with prediction data to detect distribution shifts
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Drift detection libraries
try:
    from evidently.report import Report
    from evidently.metrics import DataDriftPreset, DataQualityPreset
    from evidently import ColumnMapping
    from alibi_detect.cd import KSDrift
    from alibi_detect.utils.saving import save_detector, load_detector
except ImportError as e:
    print(f"Installing drift detection libraries: {e}")
    os.system("pip install evidently alibi-detect")
    from evidently.report import Report
    from evidently.metrics import DataDriftPreset, DataQualityPreset
    from evidently import ColumnMapping
    from alibi_detect.cd import KSDrift

def create_directories():
    """Create results directory"""
    os.makedirs('results/step6_drift', exist_ok=True)

def load_training_data():
    """Load and preprocess original training data"""
    print("📊 Loading training data...")
    
    # Load original data
    df = pd.read_csv('data/data.csv')
    
    # Load metadata to apply same preprocessing
    with open('models/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Apply same preprocessing as training
    gender_mapping = metadata['gender_mapping']
    df['gender'] = df['gender'].map(gender_mapping)
    df_clean = df.dropna()
    
    # Get features (same as training)
    feature_names = metadata['feature_names']
    X_train = df_clean[feature_names]
    
    print(f"Training data shape: {X_train.shape}")
    return X_train, feature_names

def load_prediction_data():
    """Load generated prediction data from Step 4"""
    print("📊 Loading prediction data...")
    
    try:
        # Try to load from Step 4 results
        df_pred = pd.read_csv('results/step4_api_testing/test_samples.csv')
        print(f"Loaded prediction data shape: {df_pred.shape}")
        return df_pred
    except FileNotFoundError:
        print("⚠️ Step 4 test samples not found, generating new samples...")
        return generate_prediction_samples()

def generate_prediction_samples(n_samples=100):
    """Generate prediction samples with potential drift"""
    print(f"🎲 Generating {n_samples} prediction samples with drift...")
    
    np.random.seed(42)
    
    samples = []
    for i in range(n_samples):
        # Introduce some drift patterns
        if i < n_samples // 3:
            # Younger population with higher heart rates
            age = np.random.uniform(25, 45)  # Younger than training
            thalach = np.random.uniform(150, 220)  # Higher heart rate
        elif i < 2 * n_samples // 3:
            # Older population with higher cholesterol
            age = np.random.uniform(65, 85)  # Older than training
            chol = np.random.uniform(300, 500)  # Higher cholesterol
        else:
            # Normal distribution similar to training
            age = np.random.uniform(30, 80)
            thalach = np.random.uniform(80, 200)
            chol = np.random.uniform(120, 400)
        
        sample = {
            "age": int(age),
            "gender": int(np.random.choice([0, 1])),
            "cp": int(np.random.choice([0, 1, 2, 3])),
            "trestbps": float(np.random.uniform(90, 200)),
            "chol": float(chol) if 'chol' in locals() else float(np.random.uniform(120, 400)),
            "fbs": int(np.random.choice([0, 1])),
            "restecg": int(np.random.choice([0, 1, 2])),
            "thalach": float(thalach) if 'thalach' in locals() else float(np.random.uniform(80, 200)),
            "exang": int(np.random.choice([0, 1])),
            "oldpeak": float(np.random.uniform(0, 6)),
            "slope": int(np.random.choice([0, 1, 2])),
            "ca": int(np.random.choice([0, 1, 2, 3])),
            "thal": int(np.random.choice([1, 2, 3]))
        }
        samples.append(sample)
    
    df_pred = pd.DataFrame(samples)
    df_pred.to_csv('results/step6_drift/generated_prediction_samples.csv', index=False)
    
    return df_pred

def statistical_drift_detection(X_train, X_pred, feature_names, alpha=0.05):
    """Perform statistical drift detection using various tests"""
    print("📈 Performing statistical drift detection...")
    
    drift_results = []
    
    for feature in feature_names:
        train_values = X_train[feature].values
        pred_values = X_pred[feature].values
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(train_values, pred_values)
        
        # Mann-Whitney U test (for non-parametric comparison)
        try:
            mw_stat, mw_pvalue = stats.mannwhitneyu(train_values, pred_values, alternative='two-sided')
        except ValueError:
            mw_stat, mw_pvalue = np.nan, np.nan
        
        # Anderson-Darling test
        try:
            ad_stat, ad_critical, ad_significance = stats.anderson_ksamp([train_values, pred_values])
            ad_pvalue = 1 - ad_significance if ad_stat > ad_critical[2] else ad_significance
        except:
            ad_stat, ad_pvalue = np.nan, np.nan
        
        # Effect size (Cohen's d for continuous variables)
        mean_diff = np.mean(pred_values) - np.mean(train_values)
        pooled_std = np.sqrt((np.var(train_values) + np.var(pred_values)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Determine drift
        drift_detected = ks_pvalue < alpha
        drift_severity = "High" if ks_pvalue < 0.01 else "Medium" if ks_pvalue < 0.05 else "Low"
        
        drift_results.append({
            'feature': feature,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'mw_statistic': mw_stat,
            'mw_pvalue': mw_pvalue,
            'ad_statistic': ad_stat,
            'ad_pvalue': ad_pvalue,
            'cohens_d': cohens_d,
            'drift_detected': drift_detected,
            'drift_severity': drift_severity,
            'train_mean': np.mean(train_values),
            'pred_mean': np.mean(pred_values),
            'train_std': np.std(train_values),
            'pred_std': np.std(pred_values)
        })
    
    return pd.DataFrame(drift_results)

def evidently_drift_analysis(X_train, X_pred, feature_names):
    """Use Evidently AI for comprehensive drift analysis"""
    print("🔍 Running Evidently drift analysis...")
    
    try:
        # Prepare data for Evidently
        train_df = X_train.copy()
        train_df['dataset'] = 'reference'
        
        pred_df = X_pred[feature_names].copy()  # Ensure same columns
        pred_df['dataset'] = 'current'
        
        # Column mapping
        column_mapping = ColumnMapping()
        column_mapping.numerical_features = feature_names
        
        # Create drift report
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        
        data_drift_report.run(reference_data=train_df, current_data=pred_df, column_mapping=column_mapping)
        
        # Save report
        data_drift_report.save_html('results/step6_drift/evidently_drift_report.html')
        
        # Extract key metrics
        report_dict = data_drift_report.as_dict()
        
        return {
            'evidently_available': True,
            'report_saved': True,
            'report_dict': report_dict
        }
        
    except Exception as e:
        print(f"Evidently analysis failed: {e}")
        return {
            'evidently_available': False,
            'error': str(e)
        }

def alibi_drift_detection(X_train, X_pred, feature_names):
    """Use Alibi Detect for drift detection"""
    print("🚨 Setting up Alibi drift detector...")
    
    try:
        # Convert to numpy arrays
        X_train_array = X_train[feature_names].values.astype(np.float32)
        X_pred_array = X_pred[feature_names].values.astype(np.float32)
        
        # Create KS drift detector
        drift_detector = KSDrift(X_train_array, p_val=0.05)
        
        # Detect drift
        drift_prediction = drift_detector.predict(X_pred_array)
        
        # Save detector
        save_detector(drift_detector, 'results/step6_drift/ks_drift_detector')
        
        return {
            'alibi_available': True,
            'drift_detected': bool(drift_prediction['data']['is_drift']),
            'p_value': float(drift_prediction['data']['p_val']),
            'threshold': drift_detector.p_val,
            'drift_prediction': drift_prediction
        }
        
    except Exception as e:
        print(f"Alibi detection failed: {e}")
        return {
            'alibi_available': False,
            'error': str(e)
        }

def visualize_drift(X_train, X_pred, drift_results, feature_names):
    """Create drift visualization plots"""
    print("📊 Creating drift visualizations...")
    
    # Sort features by drift severity
    drift_features = drift_results.nlargest(6, 'ks_statistic')
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.ravel()
    
    for i, (_, row) in enumerate(drift_features.iterrows()):
        if i >= 6:
            break
            
        feature = row['feature']
        
        # Distribution comparison
        ax = axes[i]
        
        train_values = X_train[feature].values
        pred_values = X_pred[feature].values
        
        ax.hist(train_values, bins=30, alpha=0.7, label='Training', color='blue', density=True)
        ax.hist(pred_values, bins=30, alpha=0.7, label='Prediction', color='red', density=True)
        
        ax.set_title(f'{feature} Distribution\nKS p-value: {row["ks_pvalue"]:.4f}', fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Training: μ={row['train_mean']:.2f}, σ={row['train_std']:.2f}\n"
        stats_text += f"Prediction: μ={row['pred_mean']:.2f}, σ={row['pred_std']:.2f}\n"
        stats_text += f"Cohen's d: {row['cohens_d']:.3f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/step6_drift/feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Drift summary heatmap
    plt.figure(figsize=(12, 8))
    
    # Create drift matrix
    drift_matrix = drift_results.set_index('feature')[['ks_pvalue', 'cohens_d', 'drift_detected']]
    
    # Convert drift_detected to numeric
    drift_matrix['drift_detected'] = drift_matrix['drift_detected'].astype(int)
    
    # Create heatmap
    sns.heatmap(drift_matrix.T, annot=True, cmap='RdYlBu_r', center=0, 
                cbar_kws={'label': 'Drift Intensity'})
    plt.title('Drift Detection Summary Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('Drift Metrics')
    plt.tight_layout()
    plt.savefig('results/step6_drift/drift_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_drift_report(drift_results, evidently_result, alibi_result):
    """Generate comprehensive drift analysis report"""
    
    report = []
    report.append("🚨 INPUT DRIFT DETECTION REPORT")
    report.append("="*50)
    report.append(f"Analysis Time: {pd.Timestamp.now()}")
    
    # Overall drift summary
    total_features = len(drift_results)
    drifted_features = len(drift_results[drift_results['drift_detected']])
    drift_percentage = (drifted_features / total_features) * 100
    
    report.append(f"\n📊 OVERALL DRIFT SUMMARY:")
    report.append(f"Total Features Analyzed: {total_features}")
    report.append(f"Features with Drift: {drifted_features}")
    report.append(f"Drift Percentage: {drift_percentage:.1f}%")
    
    # Feature-wise drift analysis
    report.append(f"\n🔍 FEATURE-WISE DRIFT ANALYSIS:")
    
    high_drift = drift_results[drift_results['drift_severity'] == 'High']
    medium_drift = drift_results[drift_results['drift_severity'] == 'Medium']
    
    if len(high_drift) > 0:
        report.append(f"\n🚨 HIGH DRIFT FEATURES:")
        for _, row in high_drift.iterrows():
            report.append(f"  • {row['feature']}: p-value={row['ks_pvalue']:.4f}, Cohen's d={row['cohens_d']:.3f}")
    
    if len(medium_drift) > 0:
        report.append(f"\n⚠️ MEDIUM DRIFT FEATURES:")
        for _, row in medium_drift.iterrows():
            report.append(f"  • {row['feature']}: p-value={row['ks_pvalue']:.4f}, Cohen's d={row['cohens_d']:.3f}")
    
    # Top 5 most drifted features
    top_drift = drift_results.nsmallest(5, 'ks_pvalue')
    report.append(f"\n📈 TOP 5 MOST DRIFTED FEATURES:")
    for i, (_, row) in enumerate(top_drift.iterrows(), 1):
        report.append(f"{i}. {row['feature']}: KS p-value={row['ks_pvalue']:.6f}")
    
    # Alibi results
    if alibi_result.get('alibi_available'):
        report.append(f"\n🤖 ALIBI DETECT RESULTS:")
        report.append(f"Overall Drift Detected: {'Yes' if alibi_result['drift_detected'] else 'No'}")
        report.append(f"P-value: {alibi_result['p_value']:.6f}")
        report.append(f"Threshold: {alibi_result['threshold']}")
    
    # Evidently results
    if evidently_result.get('evidently_available'):
        report.append(f"\n📋 EVIDENTLY AI ANALYSIS:")
        report.append(f"✅ Comprehensive drift report generated: evidently_drift_report.html")
    
    # Impact assessment
    report.append(f"\n💥 POTENTIAL IMPACT:")
    
    if drift_percentage > 50:
        report.append("🚨 CRITICAL: >50% of features show drift")
        report.append("   - Model performance may be significantly degraded")
        report.append("   - Immediate retraining recommended")
    elif drift_percentage > 25:
        report.append("⚠️ HIGH: >25% of features show drift")
        report.append("   - Monitor model performance closely")
        report.append("   - Consider retraining soon")
    elif drift_percentage > 10:
        report.append("🟡 MODERATE: >10% of features show drift")
        report.append("   - Continue monitoring")
        report.append("   - Plan for retraining")
    else:
        report.append("✅ LOW: <10% of features show drift")
        report.append("   - Current model should perform well")
        report.append("   - Continue regular monitoring")
    
    # Recommendations
    report.append(f"\n💡 RECOMMENDATIONS:")
    
    if drifted_features > 0:
        report.append("1. Investigate root causes of distribution shifts")
        report.append("2. Collect more recent training data")
        report.append("3. Consider online learning or model adaptation")
        report.append("4. Implement continuous drift monitoring")
        report.append("5. Set up automated alerts for drift detection")
    else:
        report.append("1. Continue current monitoring approach")
        report.append("2. Maintain regular drift detection schedule")
    
    report.append(f"\n📁 Generated Files:")
    report.append("- feature_distributions.png (Visual comparison)")
    report.append("- drift_heatmap.png (Summary visualization)")
    report.append("- drift_results.json (Detailed metrics)")
    if evidently_result.get('evidently_available'):
        report.append("- evidently_drift_report.html (Interactive report)")
    
    return "\n".join(report)

def save_results(drift_results, evidently_result, alibi_result, report):
    """Save all drift detection results"""
    print("💾 Saving drift detection results...")
    
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'statistical_drift': drift_results.to_dict('records'),
        'evidently_analysis': evidently_result,
        'alibi_analysis': alibi_result,
        'report': report
    }
    
    with open('results/step6_drift/drift_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save detailed drift results as CSV
    drift_results.to_csv('results/step6_drift/detailed_drift_analysis.csv', index=False)
    
    # Save report as text
    with open('results/step6_drift/drift_report.txt', 'w') as f:
        f.write(report)

def main():
    """Main execution function"""
    print("🚀 Starting Step 6: Input Drift Detection")
    print("="*50)
    
    create_directories()
    
    # Load data
    X_train, feature_names = load_training_data()
    X_pred = load_prediction_data()
    
    print(f"Training data: {X_train.shape}")
    print(f"Prediction data: {X_pred.shape}")
    
    # Statistical drift detection
    drift_results = statistical_drift_detection(X_train, X_pred, feature_names)
    
    # Advanced drift analysis
    evidently_result = evidently_drift_analysis(X_train, X_pred, feature_names)
    alibi_result = alibi_drift_detection(X_train, X_pred, feature_names)
    
    # Visualizations
    visualize_drift(X_train, X_pred, drift_results, feature_names)
    
    # Generate report
    report = generate_drift_report(drift_results, evidently_result, alibi_result)
    print(report)
    
    # Save results
    save_results(drift_results, evidently_result, alibi_result, report)
    
    print("\n✅ Step 6: Drift detection completed!")
    print("📊 Results saved to: results/step6_drift/")

if __name__ == "__main__":
    main()
