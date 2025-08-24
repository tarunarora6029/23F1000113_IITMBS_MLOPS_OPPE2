# Heart Disease MLOps Pipeline

Complete MLOps pipeline for heart disease prediction with explainability, fairness, deployment, monitoring, and security testing.

## 🚀 Overview

This project implements a comprehensive MLOps pipeline that includes:

1. **Explainability Analysis** - LIME & SHAP for model interpretability
2. **Fairness Testing** - Gender bias analysis with fairlearn
3. **Dockerized API Deployment** - FastAPI on GCP with Kubernetes
4. **Observability** - Logging, monitoring, and sample predictions
5. **Performance Testing** - High concurrency testing with wrk
6. **Drift Detection** - Input data distribution monitoring
7. **Security Testing** - Data poisoning attack simulation

## 📊 Dataset

UCI Heart Disease Dataset with features:
- `age`: Patient age
- `gender`: 0=male, 1=female  
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure
- `chol`: Cholesterol level
- `fbs`: Fasting blood sugar > 120 mg/dl
- `restecg`: Resting ECG results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina
- `oldpeak`: ST depression from exercise
- `slope`: ST segment slope
- `ca`: Major vessels count
- `thal`: Thalassemia type

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GitHub Repo   │───▶│ GitHub Actions  │───▶│   GCP/GKE      │
│                 │    │    CI/CD        │    │   Deployment   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Data & Models   │    │ ML Pipeline     │    │ K8s Services    │
│ • data.csv      │    │ • Training      │    │ • API Pods      │
│ • Trained Model │    │ • Testing       │    │ • Auto-scaling  │
│ • Preprocessors │    │ • Analysis      │    │ • Load Balancer │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Setup

### Prerequisites

- Google Cloud Platform account
- GKE cluster configured
- GitHub repository with secrets:
  - `GCP_PROJECT_ID`
  - `GCP_SERVICE_ACCOUNT_KEY`

### Repository Structure

```
heart-disease-mlops/
├── .github/workflows/
│   └── mlops-pipeline.yml      # Main CI/CD workflow
├── scripts/                    # All step scripts
│   ├── step1_explainability.py
│   ├── step2_fairness.py
│   ├── step4_api_testing.py
│   ├── step5_performance.sh
│   ├── step6_drift.py
│   ├── step7_security.py
│   └── deploy.sh
├── src/
│   └── api.py                  # FastAPI application
├── k8s/                        # Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── data/
│   └── data.csv               # Heart disease dataset
├── models/                    # Saved models (generated)
├── results/                   # Analysis results (generated)
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 Pipeline Steps

### Step 1: Explainability Analysis

**Script**: `scripts/step1_explainability.py`

- Trains logistic regression model with hyperparameter tuning
- Generates SHAP visualizations for global interpretability
- Creates LIME explanations for individual predictions
- Provides plain English explanations of key factors

**Outputs**:
- Model artifacts (`models/`)
- Feature importance plots
- SHAP summary plots  
- LIME HTML explanations
- Plain English summary

### Step 2: Fairness Testing

**Script**: `scripts/step2_fairness.py`

- Uses fairlearn for bias analysis with gender as sensitive attribute
- Tests demographic parity and equalized odds
- Applies post-processing fairness mitigation
- Generates comprehensive fairness assessment

**Outputs**:
- Performance by gender visualizations
- Fairness metrics analysis
- Bias mitigation results
- Fairness recommendations

### Step 3: Containerized Deployment

**Files**: `Dockerfile`, `src/api.py`, `k8s/`

- Builds Docker image with FastAPI application
- Deploys to GKE with auto-scaling (max 3 pods)
- Includes health checks and monitoring endpoints
- Structured logging and Prometheus metrics

**Features**:
- `/predict` - Single predictions with explainability
- `/predict/batch` - Batch predictions
- `/health` - Health monitoring
- `/metrics` - Prometheus metrics

### Step 4: API Testing & Observability

**Script**: `scripts/step4_api_testing.py`

- Generates 100 random test samples
- Tests all API endpoints
- Performs individual and batch predictions
- Analyzes response times and success rates
- Validates observability features

**Outputs**:
- API performance metrics
- Success rate analysis
- Response time distributions
- Observability report

### Step 5: Performance Testing

**Script**: `scripts/step5_performance.sh`

- High concurrency testing with `wrk`
- Tests various load levels (2-16 threads, 10-300 connections)
- Measures throughput, latency, and timeout analysis
- Identifies performance bottlenecks

**Test Scenarios**:
- Baseline: 2 threads, 10 connections
- Moderate: 4 threads, 50 connections
- High Load: 8 threads, 100 connections
- Stress Test: 12 threads, 200 connections
- Burst Test: 16 threads, 300 connections

### Step 6: Drift Detection

**Script**: `scripts/step6_drift.py`

- Compares training vs prediction data distributions
- Uses statistical tests (KS-test, Anderson-Darling)
- Implements Evidently AI for comprehensive analysis
- Alibi Detect for drift monitoring
- Visualizes distribution shifts

**Outputs**:
- Statistical drift analysis
- Distribution comparison plots
- Interactive drift reports
- Drift severity assessment

### Step 7: Security Testing

**Script**: `scripts/step7_security.py`

- Simulates data poisoning attacks (label interchange)
- Tests various poisoning levels (5%-25%)
- Analyzes performance degradation
- Compares feature importance changes
- Attempts poison sample detection

**Outputs**:
- Performance degradation analysis
- Attack impact visualizations
- Security vulnerability assessment
- Mitigation recommendations

## 🔄 CI/CD Workflow

The pipeline runs automatically on push to `main` or `develop`:

1. **Setup & Dependencies** - Install required packages
2. **Parallel Execution**:
   - Steps 1-2 run sequentially (model training → fairness)
   - Step 3 deploys to GCP/GKE
   - Steps 4-7 run in parallel using deployed API
3. **Final Report** - Aggregates all results

## 📊 Results & Monitoring

Each step generates detailed results:

- **JSON files** - Structured data and metrics
- **Visualizations** - Charts and plots
- **HTML reports** - Interactive analysis
- **Text summaries** - Human-readable insights

### Key Metrics Tracked

- **Model Performance**: Accuracy, Precision, Recall, F1
- **Fairness**: Demographic parity, Equalized odds
- **API Performance**: Response times, Throughput, Success rates
- **Drift**: Statistical tests, Distribution changes
- **Security**: Attack impact, Vulnerability assessment

## 🛡️ Security & Best Practices

- **Non-root container** execution
- **Resource limits** on K8s pods
- **Health checks** and liveness probes
- **Structured logging** for observability
- **Secrets management** via GitHub Secrets
- **Input validation** on API endpoints

## 📈 Monitoring & Alerting

- **Prometheus metrics** for performance monitoring
- **Structured logging** for troubleshooting  
- **Health endpoints** for service monitoring
- **Auto-scaling** based on CPU/memory usage
- **Drift detection** for model degradation

## 🚨 Alerts & Thresholds

- **Fairness**: Bias > 20% triggers alerts
- **Performance**: Response time > 2s P95
- **Drift**: >25% features showing drift
- **Security**: >20% performance degradation

## 🔧 Usage

### Automatic (CI/CD)
Push to main branch triggers full pipeline:

```bash
git add .
git commit -m "Deploy MLOps pipeline"
git push origin main
```

### Manual Execution
Run individual steps locally:

```bash
# Step 1: Explainability
python scripts/step1_explainability.py

# Step 2: Fairness  
python scripts/step2_fairness.py

# Step 6: Drift Detection
python scripts/step6_drift.py

# Step 7: Security Testing
python scripts/step7_security.py
```

## 📋 Requirements

See `requirements.txt` for complete dependencies.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test locally
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 📞 Support

For questions or issues, please create a GitHub issue with:
- Error messages
- Steps to reproduce
- Environment details
