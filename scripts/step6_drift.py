#!/usr/bin/env python3
"""
Step 6: Input Drift Detection
Using Evidently AI only for comprehensive drift analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Evidently library
try:
    from evidently.dashboard import Dashboard
    from evidently.dashboard.tabs import DataDriftTab
    from evidently.model_profile import Profile
    from evidently.model_profile.sections import DataDriftProfileSection
except ImportError as e:
    print(f"Installing evidently: {e}")
    os.system("pip install evidently==0.2.8")
    from evidently.dashboard import Dashboard
    from evidently.dashboard.tabs import DataDriftTab
    from evidently.model_profile import Profile
    from evidently.model_profile.sections import DataDriftProfileSection

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

def evidently_drift_analysis(X_train, X_pred, feature_names):
    """Use Evidently AI for comprehensive drift analysis"""
    print("🔍 Running Evidently drift analysis...")
    
    try:
        # Prepare data for Evidently
        reference_data = X_train.copy()
        current_data = X_pred[feature_names].copy()  # Ensure same columns
        
        print(f"Reference data shape: {reference_data.shape}")
        print(f"Current data shape: {current_data.shape}")
        
        # Create drift dashboard
        data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
        data_drift_dashboard.calculate(reference_data, current_data)
        
        # Save interactive dashboard
        dashboard_path = 'results/step6_drift/evidently_drift_dashboard.html'
        data_drift_dashboard.save(dashboard_path)
        print(f"✅ Dashboard saved to: {dashboard_path}")
        
        # Create profile for extracting metrics
        data_drift_profile = Profile(sections=[DataDriftProfileSection()])
        data_drift_profile.calculate(reference_data, current_data)
        
        # Save profile JSON
        profile_json = data_drift_profile.json()
        profile_path = 'results/step6_drift/evidently_drift_profile.json'
        with open(profile_path, 'w') as f:
            f.write(profile_json)
        print(f"✅ Profile saved to: {profile_path}")
        
        # Parse profile for key metrics
        profile_dict = json.loads(profile_json)
        
        # Extract drift metrics from profile
        drift_section = profile_dict.get('data_drift', {})
        drift_data = drift_section.get('data', {})
        drift_metrics = drift_data.get('metrics', {})
        
        drift_info = {
            'evidently_available': True,
            'dashboard_saved': True,
            'profile_saved': True,
            'n_features': drift_metrics.get('n_features', len(feature_names)),
            'n_drifted_features': drift_metrics.get('n_drifted_features', 0),
            'share_drifted_features': drift_metrics.get('share_drifted_features', 0.0),
            'dataset_drift': drift_metrics.get('dataset_drift', False),
            'drift_by_columns': drift_data.get('drift_by_columns', {}),
            'feature_names': feature_names
        }
        
        print(f"📊 Drift Analysis Results:")
        print(f"   Total features: {drift_info['n_features']}")
        print(f"   Drifted features: {drift_info['n_drifted_features']}")
        print(f"   Drift percentage: {drift_info['share_drifted_features']:.1%}")
        print(f"   Dataset drift detected: {drift_info['dataset_drift']}")
        
        return drift_info
        
    except Exception as e:
        print(f"❌ Evidently analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'evidently_available': False,
            'error': str(e)
        }

def generate_drift_report(drift_info):
    """Generate comprehensive drift analysis report using Evidently results"""
    
    report = []
    report.append("🚨 INPUT DRIFT DETECTION REPORT")
    report.append("="*50)
    report.append(f"Analysis Time: {pd.Timestamp.now()}")
    report.append("Tool: Evidently AI")
    
    if not drift_info.get('evidently_available', False):
        report.append(f"\n❌ ANALYSIS FAILED:")
        report.append(f"Error: {drift_info.get('error', 'Unknown error')}")
        return "\n".join(report)
    
    # Overall drift summary
    total_features = drift_info.get('n_features', 0)
    drifted_features = drift_info.get('n_drifted_features', 0)
    drift_percentage = drift_info.get('share_drifted_features', 0.0) * 100
    dataset_drift = drift_info.get('dataset_drift', False)
    
    report.append(f"\n📊 OVERALL DRIFT SUMMARY:")
    report.append(f"Total Features Analyzed: {total_features}")
    report.append(f"Features with Drift: {drifted_features}")
    report.append(f"Drift Percentage: {drift_percentage:.1f}%")
    report.append(f"Dataset-level Drift: {'Yes' if dataset_drift else 'No'}")
    
    # Feature-wise drift analysis
    drift_by_columns = drift_info.get('drift_by_columns', {})
    if drift_by_columns:
        report.append(f"\n🔍 FEATURE-WISE DRIFT ANALYSIS:")
        
        drifted_features_list = []
        stable_features_list = []
        
        for feature, drift_data in drift_by_columns.items():
            if isinstance(drift_data, dict) and drift_data.get('drift_detected', False):
                p_value = drift_data.get('stattest_threshold', 'N/A')
                drifted_features_list.append(f"  • {feature}: p-value threshold {p_value}")
            else:
                stable_features_list.append(f"  • {feature}")
        
        if drifted_features_list:
            report.append(f"\n🚨 FEATURES WITH DRIFT:")
            report.extend(drifted_features_list)
        
        if stable_features_list:
            report.append(f"\n✅ STABLE FEATURES:")
            report.extend(stable_features_list[:5])  # Show first 5 to keep report manageable
            if len(stable_features_list) > 5:
                report.append(f"  ... and {len(stable_features_list) - 5} more")
    
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
    
    if drift_percentage > 0:
        report.append("1. Investigate root causes of distribution shifts")
        report.append("2. Collect more recent training data")
        report.append("3. Consider online learning or model adaptation")
        report.append("4. Implement continuous drift monitoring")
        report.append("5. Set up automated alerts for drift detection")
    else:
        report.append("1. Continue current monitoring approach")
        report.append("2. Maintain regular drift detection schedule")
    
    report.append(f"\n📁 Generated Files:")
    report.append("- evidently_drift_dashboard.html (Interactive dashboard)")
    report.append("- evidently_drift_profile.json (Detailed metrics)")
    report.append("- drift_results.json (Summary results)")
    report.append("- drift_report.txt (This report)")
    
    return "\n".join(report)

def save_results(drift_info, report):
    """Save all drift detection results"""
    print("💾 Saving drift detection results...")
    
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'tool': 'Evidently AI',
        'evidently_analysis': drift_info,
        'report': report
    }
    
    with open('results/step6_drift/drift_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save report as text
    with open('results/step6_drift/drift_report.txt', 'w') as f:
        f.write(report)

def main():
    """Main execution function"""
    print("🚀 Starting Step 6: Input Drift Detection with Evidently AI")
    print("="*60)
    
    create_directories()
    
    # Load data
    X_train, feature_names = load_training_data()
    X_pred = load_prediction_data()
    
    print(f"Training data: {X_train.shape}")
    print(f"Prediction data: {X_pred.shape}")
    
    # Run Evidently drift analysis
    drift_info = evidently_drift_analysis(X_train, X_pred, feature_names)
    
    # Generate report
    report = generate_drift_report(drift_info)
    print("\n" + "="*60)
    print(report)
    
    # Save results
    save_results(drift_info, report)
    
    print("\n✅ Step 6: Drift detection completed!")
    print("📊 Results saved to: results/step6_drift/")
    print("🌐 Open evidently_drift_dashboard.html in browser for interactive analysis")

if __name__ == "__main__":
    main()
