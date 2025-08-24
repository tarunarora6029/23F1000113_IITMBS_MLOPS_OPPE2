#!/usr/bin/env python3
"""
Step 1: Model Explainability Analysis
Automated script for CI/CD pipeline
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For CI/CD
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Explainability libraries
try:
    import lime
    import lime.lime_tabular
    import shap
except ImportError as e:
    print(f"Installing missing library: {e}")
    os.system("pip install lime shap")
    import lime
    import lime.lime_tabular
    import shap

def create_directories():
    """Create necessary directories"""
    os.makedirs('results/step1_explainability', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess heart disease data"""
    print("📊 Step 1: Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv('data/data.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Gender mapping
    gender_mapping = {'male': 0, 'female': 1}
    df['gender'] = df['gender'].map(gender_mapping)
    
    # Clean data
    df_clean = df.dropna()
    print(f"Clean dataset shape: {df_clean.shape}")
    
    # Features and target
    X = df_clean.drop(['target', 'sno'], axis=1)
    y = df_clean['target']
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le, gender_mapping, list(X.columns)

def train_model(X, y):
    """Train logistic regression model"""
    print("🔧 Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning
    param_grid = {
        "C": np.logspace(-4, 4, 10),
        "solver": ["liblinear"],
        "max_iter": [1000]
    }
    
    grid_search = RandomizedSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=5,
        n_iter=10,
        random_state=42,
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    model = grid_search.best_estimator_
    
    # Performance
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"Best params: {grid_search.best_params_}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test

def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance"""
    print("📈 Analyzing feature importance...")
    
    coeffs = model.coef_[0]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coeffs,
        'abs_coefficient': np.abs(coeffs)
    }).sort_values('abs_coefficient', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    sns.barplot(data=importance_df.head(10), x='abs_coefficient', y='feature')
    plt.title('Top 10 Features by Importance')
    
    plt.subplot(2, 1, 2)
    colors = ['red' if x < 0 else 'green' for x in importance_df.head(10)['coefficient']]
    sns.barplot(data=importance_df.head(10), x='coefficient', y='feature', palette=colors)
    plt.title('Feature Coefficients (Green=Risk+, Red=Risk-)')
    
    plt.tight_layout()
    plt.savefig('results/step1_explainability/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return importance_df

def shap_analysis(model, X_train_scaled, X_test_scaled, feature_names):
    """SHAP analysis"""
    print("🔍 SHAP analysis...")
    
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled[:50])  # Limit for CI/CD
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled[:50], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('results/step1_explainability/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Feature importance
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test_scaled[:50], feature_names=feature_names, 
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('results/step1_explainability/shap_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return shap_values

def lime_analysis(model, X_train_scaled, X_test_scaled, feature_names, n_samples=3):
    """LIME analysis for individual predictions"""
    print(f"🍋 LIME analysis on {n_samples} samples...")
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_scaled,
        feature_names=feature_names,
        class_names=['No Disease', 'Disease'],
        mode='classification'
    )
    
    lime_results = []
    for i in range(min(n_samples, len(X_test_scaled))):
        explanation = explainer.explain_instance(
            X_test_scaled[i],
            model.predict_proba,
            num_features=len(feature_names)
        )
        
        explanation.save_to_file(f'results/step1_explainability/lime_sample_{i+1}.html')
        
        pred = model.predict(X_test_scaled[i].reshape(1, -1))[0]
        prob = model.predict_proba(X_test_scaled[i].reshape(1, -1))[0]
        
        lime_results.append({
            'sample': i+1,
            'prediction': int(pred),
            'probability': prob.tolist(),
            'top_features': explanation.as_list()[:5]
        })
    
    return lime_results

def generate_plain_english_summary(importance_df):
    """Generate plain English explanation"""
    
    feature_explanations = {
        'age': 'Patient age (years)',
        'gender': 'Gender (0=male, 1=female)', 
        'cp': 'Chest pain type (0-3 scale)',
        'trestbps': 'Resting blood pressure (mmHg)',
        'chol': 'Cholesterol level (mg/dl)',
        'fbs': 'Fasting blood sugar >120 (1=yes)',
        'restecg': 'Resting ECG results (0-2 scale)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1=yes)',
        'oldpeak': 'ST depression from exercise',
        'slope': 'Exercise ST segment slope',
        'ca': 'Major vessels colored (0-3)',
        'thal': 'Thalassemia type (1-3)'
    }
    
    summary = []
    summary.append("🔍 HEART DISEASE PREDICTION FACTORS:")
    summary.append("="*50)
    
    top_5 = importance_df.head(5)
    
    for idx, row in top_5.iterrows():
        feature = row['feature']
        coef = row['coefficient']
        impact = "INCREASES" if coef > 0 else "DECREASES"
        
        desc = feature_explanations.get(feature, feature)
        summary.append(f"\n{idx+1}. {desc.upper()}")
        summary.append(f"   Impact: {impact} heart disease risk")
        summary.append(f"   Strength: {abs(coef):.3f}")
    
    summary.append("\n💡 KEY INSIGHTS:")
    summary.append("- Model uses clinical measurements and symptoms")
    summary.append("- Higher coefficients = stronger predictive power") 
    summary.append("- Individual cases may vary based on patient profile")
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    # Save summary
    with open('results/step1_explainability/plain_english_summary.txt', 'w') as f:
        f.write(summary_text)
    
    return summary_text

def save_results(model, scaler, label_encoder, feature_names, gender_mapping, 
                importance_df, lime_results):
    """Save all models and results"""
    print("💾 Saving models and results...")
    
    # Save models
    joblib.dump(model, 'models/heart_disease_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl') 
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'gender_mapping': gender_mapping,
        'model_type': 'LogisticRegression',
        'target_classes': label_encoder.classes_.tolist()
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save results
    results = {
        'feature_importance': importance_df.to_dict('records'),
        'lime_results': lime_results
    }
    
    with open('results/step1_explainability/results.json', 'w') as f:
        json.dump(results, f, indent=2)

def main():
    """Main execution function"""
    print("🚀 Starting Step 1: Explainability Analysis")
    print("="*50)
    
    create_directories()
    
    # Load and preprocess
    X, y, label_encoder, gender_mapping, feature_names = load_and_preprocess_data()
    
    # Train model
    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test = train_model(X, y)
    
    # Analyze explainability
    importance_df = analyze_feature_importance(model, feature_names)
    shap_values = shap_analysis(model, X_train_scaled, X_test_scaled, feature_names)
    lime_results = lime_analysis(model, X_train_scaled, X_test_scaled, feature_names)
    
    # Generate summary
    plain_english = generate_plain_english_summary(importance_df)
    
    # Save everything
    save_results(model, scaler, label_encoder, feature_names, gender_mapping,
                importance_df, lime_results)
    
    print("\n✅ Step 1 completed successfully!")
    print(f"📊 Results saved to: results/step1_explainability/")
    print(f"🤖 Models saved to: models/")
    
if __name__ == "__main__":
    main()
