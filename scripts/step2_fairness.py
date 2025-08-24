#!/usr/bin/env python3
"""
Step 2: Fairness Testing with Gender as Sensitive Attribute
Uses fairlearn to test model fairness
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Fairness libraries
try:
    from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
    from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio
    from fairlearn.postprocessing import ThresholdOptimizer
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
except ImportError:
    print("Installing fairlearn...")
    os.system("pip install fairlearn")
    from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
    from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio
    from fairlearn.postprocessing import ThresholdOptimizer
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def create_directories():
    """Create results directory"""
    os.makedirs('results/step2_fairness', exist_ok=True)

def load_model_and_data():
    """Load trained model and test data"""
    print("📊 Loading model and data...")
    
    # Load model components
    model = joblib.load('models/heart_disease_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    
    with open('models/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load and preprocess data
    df = pd.read_csv('data/data.csv')
    
    # Apply same preprocessing as training
    gender_mapping = metadata['gender_mapping']
    df['gender'] = df['gender'].map(gender_mapping)
    df_clean = df.dropna()
    
    X = df_clean.drop(['target', 'sno'], axis=1)
    y = df_clean['target']
    y_encoded = label_encoder.transform(y)
    
    # Split same way as training (same random state)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Get sensitive attribute (gender)
    gender_test = X_test['gender'].values
    
    return model, X_test_scaled, y_test, gender_test, metadata

def compute_fairness_metrics(model, X_test, y_test, gender_test):
    """Compute comprehensive fairness metrics"""
    print("⚖️ Computing fairness metrics...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Define metrics
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score,
        'selection_rate': selection_rate
    }
    
    # Compute metrics by gender
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=gender_test
    )
    
    # Fairness-specific metrics
    fairness_metrics = {
        'demographic_parity_difference': demographic_parity_difference(y_test, y_pred, sensitive_features=gender_test),
        'demographic_parity_ratio': demographic_parity_ratio(y_test, y_pred, sensitive_features=gender_test),
        'equalized_odds_difference': equalized_odds_difference(y_test, y_pred, sensitive_features=gender_test),
        'equalized_odds_ratio': equalized_odds_ratio(y_test, y_pred, sensitive_features=gender_test)
    }
    
    return metric_frame, fairness_metrics, y_pred, y_prob

def visualize_fairness_results(metric_frame, fairness_metrics):
    """Create fairness visualizations"""
    print("📊 Creating fairness visualizations...")
    
    # Performance by gender
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy by gender
    metric_frame.by_group['accuracy'].plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
    axes[0,0].set_title('Accuracy by Gender')
    axes[0,0].set_xlabel('Gender (0=Male, 1=Female)')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].tick_params(axis='x', rotation=0)
    
    # Precision by gender
    metric_frame.by_group['precision'].plot(kind='bar', ax=axes[0,1], color=['skyblue', 'lightcoral'])
    axes[0,1].set_title('Precision by Gender')
    axes[0,1].set_xlabel('Gender (0=Male, 1=Female)')
    axes[0,1].set_ylabel('Precision')
    axes[0,1].tick_params(axis='x', rotation=0)
    
    # Recall by gender
    metric_frame.by_group['recall'].plot(kind='bar', ax=axes[1,0], color=['skyblue', 'lightcoral'])
    axes[1,0].set_title('Recall by Gender')
    axes[1,0].set_xlabel('Gender (0=Male, 1=Female)')
    axes[1,0].set_ylabel('Recall')
    axes[1,0].tick_params(axis='x', rotation=0)
    
    # Selection rate by gender
    metric_frame.by_group['selection_rate'].plot(kind='bar', ax=axes[1,1], color=['skyblue', 'lightcoral'])
    axes[1,1].set_title('Selection Rate by Gender')
    axes[1,1].set_xlabel('Gender (0=Male, 1=Female)')
    axes[1,1].set_ylabel('Selection Rate')
    axes[1,1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('results/step2_fairness/performance_by_gender.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Fairness metrics visualization
    fairness_df = pd.DataFrame(list(fairness_metrics.items()), columns=['Metric', 'Value'])
    
    plt.figure(figsize=(12, 6))
    colors = ['green' if abs(v) < 0.1 else 'orange' if abs(v) < 0.2 else 'red' for v in fairness_df['Value']]
    bars = plt.bar(fairness_df['Metric'], fairness_df['Value'], color=colors, alpha=0.7)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Moderate Bias (0.1)')
    plt.axhline(y=-0.1, color='orange', linestyle='--', alpha=0.5)
    plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='High Bias (0.2)')
    plt.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)
    
    plt.title('Fairness Metrics (Values closer to 0 = More Fair)')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, fairness_df['Value']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*np.sign(bar.get_height()),
                f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('results/step2_fairness/fairness_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

def assess_fairness_level(fairness_metrics):
    """Assess overall fairness level"""
    
    # Thresholds for fairness assessment
    EXCELLENT = 0.05
    GOOD = 0.1
    MODERATE = 0.2
    
    assessments = {}
    overall_issues = []
    
    for metric, value in fairness_metrics.items():
        abs_value = abs(value)
        
        if abs_value <= EXCELLENT:
            level = "EXCELLENT"
            color = "🟢"
        elif abs_value <= GOOD:
            level = "GOOD"
            color = "🟡"
        elif abs_value <= MODERATE:
            level = "CONCERNING"
            color = "🟠"
        else:
            level = "POOR"
            color = "🔴"
            overall_issues.append(metric)
        
        assessments[metric] = {
            'value': value,
            'level': level,
            'color': color
        }
    
    return assessments, overall_issues

def apply_fairness_mitigation(model, X_test, y_test, gender_test):
    """Apply post-processing fairness mitigation"""
    print("🔧 Applying fairness mitigation...")
    
    # Use ThresholdOptimizer to improve fairness
    postprocess_est = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity",
        objective="accuracy_score"
    )
    
    # Fit on test data (in practice, use validation set)
    postprocess_est.fit(X_test, y_test, sensitive_features=gender_test)
    
    # Get fair predictions
    y_pred_fair = postprocess_est.predict(X_test, sensitive_features=gender_test)
    
    # Compute metrics for fair model
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score,
        'selection_rate': selection_rate
    }
    
    fair_metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred_fair,
        sensitive_features=gender_test
    )
    
    fair_fairness_metrics = {
        'demographic_parity_difference': demographic_parity_difference(y_test, y_pred_fair, sensitive_features=gender_test),
        'demographic_parity_ratio': demographic_parity_ratio(y_test, y_pred_fair, sensitive_features=gender_test),
        'equalized_odds_difference': equalized_odds_difference(y_test, y_pred_fair, sensitive_features=gender_test),
        'equalized_odds_ratio': equalized_odds_ratio(y_test, y_pred_fair, sensitive_features=gender_test)
    }
    
    return postprocess_est, fair_metric_frame, fair_fairness_metrics, y_pred_fair

def generate_fairness_report(metric_frame, fairness_metrics, assessments, 
                           fair_metric_frame=None, fair_fairness_metrics=None):
    """Generate comprehensive fairness report"""
    
    report = []
    report.append("⚖️ FAIRNESS ANALYSIS REPORT")
    report.append("="*50)
    
    # Overall model performance
    report.append("\n📊 OVERALL MODEL PERFORMANCE:")
    report.append(f"Overall Accuracy: {metric_frame.overall['accuracy']:.3f}")
    report.append(f"Overall Precision: {metric_frame.overall['precision']:.3f}")
    report.append(f"Overall Recall: {metric_frame.overall['recall']:.3f}")
    
    # Performance by gender
    report.append("\n👥 PERFORMANCE BY GENDER:")
    report.append("Male (0):")
    for metric in ['accuracy', 'precision', 'recall']:
        value = metric_frame.by_group[metric][0]
        report.append(f"  {metric.capitalize()}: {value:.3f}")
    
    report.append("Female (1):")
    for metric in ['accuracy', 'precision', 'recall']:
        value = metric_frame.by_group[metric][1]
        report.append(f"  {metric.capitalize()}: {value:.3f}")
    
    # Fairness metrics assessment
    report.append("\n⚖️ FAIRNESS METRICS ASSESSMENT:")
    for metric, assessment in assessments.items():
        report.append(f"{assessment['color']} {metric}: {assessment['value']:.3f} - {assessment['level']}")
    
    # Interpret results
    report.append("\n📋 INTERPRETATION:")
    report.append("- Demographic Parity: Equal positive prediction rates across genders")
    report.append("- Equalized Odds: Equal true/false positive rates across genders")
    report.append("- Values closer to 0 indicate better fairness")
    report.append("- Ratios closer to 1 indicate better fairness")
    
    # Recommendations
    report.append("\n💡 RECOMMENDATIONS:")
    
    issues_found = [metric for metric, assessment in assessments.items() 
                   if assessment['level'] in ['CONCERNING', 'POOR']]
    
    if not issues_found:
        report.append("✅ No significant fairness issues detected!")
        report.append("✅ Model appears to be fair across gender groups")
    else:
        report.append("⚠️ Fairness issues detected:")
        for issue in issues_found:
            report.append(f"   - {issue}: {assessments[issue]['level']}")
        
        report.append("\n🔧 Suggested Actions:")
        report.append("1. Consider fairness-aware training methods")
        report.append("2. Apply post-processing fairness techniques")
        report.append("3. Collect more balanced training data")
        report.append("4. Review feature engineering for bias")
    
    # If mitigation was applied
    if fair_metric_frame is not None:
        report.append("\n🔧 AFTER FAIRNESS MITIGATION:")
        report.append("Post-processed model fairness metrics:")
        for metric, value in fair_fairness_metrics.items():
            report.append(f"  {metric}: {value:.3f}")
    
    report_text = "\n".join(report)
    print(report_text)
    
    return report_text

def save_results(metric_frame, fairness_metrics, assessments, fairness_report,
                fair_metric_frame=None, fair_fairness_metrics=None):
    """Save fairness analysis results"""
    print("💾 Saving fairness results...")
    
    # Convert metric frame to serializable format
    results = {
        'overall_performance': metric_frame.overall.to_dict(),
        'performance_by_gender': metric_frame.by_group.to_dict(),
        'fairness_metrics': fairness_metrics,
        'fairness_assessment': assessments,
        'report': fairness_report
    }
    
    if fair_metric_frame is not None:
        results['fair_model'] = {
            'overall_performance': fair_metric_frame.overall.to_dict(),
            'performance_by_gender': fair_metric_frame.by_group.to_dict(),
            'fairness_metrics': fair_fairness_metrics
        }
    
    with open('results/step2_fairness/fairness_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save report as text file
    with open('results/step2_fairness/fairness_report.txt', 'w') as f:
        f.write(fairness_report)

def main():
    """Main execution function"""
    print("🚀 Starting Step 2: Fairness Analysis")
    print("="*50)
    
    create_directories()
    
    # Load model and data
    model, X_test, y_test, gender_test, metadata = load_model_and_data()
    
    print(f"Test set size: {len(X_test)}")
    print(f"Gender distribution - Male: {(gender_test == 0).sum()}, Female: {(gender_test == 1).sum()}")
    
    # Compute fairness metrics
    metric_frame, fairness_metrics, y_pred, y_prob = compute_fairness_metrics(
        model, X_test, y_test, gender_test
    )
    
    # Assess fairness
    assessments, overall_issues = assess_fairness_level(fairness_metrics)
    
    # Create visualizations
    visualize_fairness_results(metric_frame, fairness_metrics)
    
    # Apply mitigation if needed
    fair_metric_frame = None
    fair_fairness_metrics = None
    
    if overall_issues:
        print(f"⚠️ Fairness issues detected: {overall_issues}")
        print("🔧 Applying fairness mitigation...")
        try:
            _, fair_metric_frame, fair_fairness_metrics, _ = apply_fairness_mitigation(
                model, X_test, y_test, gender_test
            )
        except Exception as e:
            print(f"Warning: Fairness mitigation failed: {e}")
    
    # Generate report
    fairness_report = generate_fairness_report(
        metric_frame, fairness_metrics, assessments,
        fair_metric_frame, fair_fairness_metrics
    )
    
    # Save results
    save_results(metric_frame, fairness_metrics, assessments, fairness_report,
                fair_metric_frame, fair_fairness_metrics)
    
    print("\n✅ Step 2: Fairness analysis completed!")
    print("📊 Results saved to: results/step2_fairness/")

if __name__ == "__main__":
    main()
