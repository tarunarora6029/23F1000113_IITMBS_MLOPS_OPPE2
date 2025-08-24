#!/usr/bin/env python3
"""
Step 4: API Testing with Observability
Generate random samples and test deployed API
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import os
import subprocess
from datetime import datetime
from typing import Dict, List

def create_directories():
    """Create results directory"""
    os.makedirs('results/step4_api_testing', exist_ok=True)

def get_service_endpoint():
    """Get Kubernetes service endpoint"""
    try:
        # Get external IP from service
        result = subprocess.run([
            'kubectl', 'get', 'service', 'heart-disease-service', 
            '-o', 'jsonpath={.status.loadBalancer.ingress[0].ip}'
        ], capture_output=True, text=True, check=True)
        
        external_ip = result.stdout.strip()
        if external_ip:
            return f"http://{external_ip}"
        
        # Fallback: try to get NodePort
        result = subprocess.run([
            'kubectl', 'get', 'service', 'heart-disease-service',
            '-o', 'jsonpath={.spec.ports[0].nodePort}'
        ], capture_output=True, text=True, check=True)
        
        node_port = result.stdout.strip()
        if node_port:
            # Get node IP
            result = subprocess.run([
                'kubectl', 'get', 'nodes', 
                '-o', 'jsonpath={.items[0].status.addresses[0].address}'
            ], capture_output=True, text=True, check=True)
            
            node_ip = result.stdout.strip()
            return f"http://{node_ip}:{node_port}"
            
    except Exception as e:
        print(f"Error getting service endpoint: {e}")
        return None

def wait_for_service(endpoint, timeout=300):
    """Wait for service to become available"""
    print(f"⏳ Waiting for service at {endpoint}/health...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{endpoint}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Service is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(5)
    
    print("❌ Service not ready within timeout")
    return False

def generate_random_samples(n_samples=100, seed=42):
    """Generate random heart disease samples for testing"""
    print(f"🎲 Generating {n_samples} random samples...")
    
    np.random.seed(seed)
    
    samples = []
    for i in range(n_samples):
        sample = {
            "age": int(np.random.uniform(30, 80)),
            "gender": int(np.random.choice([0, 1])),
            "cp": int(np.random.choice([0, 1, 2, 3])),
            "trestbps": float(np.random.uniform(90, 200)),
            "chol": float(np.random.uniform(120, 400)),
            "fbs": int(np.random.choice([0, 1])),
            "restecg": int(np.random.choice([0, 1, 2])),
            "thalach": float(np.random.uniform(80, 200)),
            "exang": int(np.random.choice([0, 1])),
            "oldpeak": float(np.random.uniform(0, 6)),
            "slope": int(np.random.choice([0, 1, 2])),
            "ca": int(np.random.choice([0, 1, 2, 3])),
            "thal": int(np.random.choice([1, 2, 3]))
        }
        samples.append(sample)
    
    return samples

def test_api_endpoints(endpoint):
    """Test all API endpoints"""
    print("🔍 Testing API endpoints...")
    
    test_results = []
    
    # Test root endpoint
    try:
        response = requests.get(f"{endpoint}/")
        test_results.append({
            'endpoint': '/',
            'method': 'GET',
            'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds(),
            'success': response.status_code == 200
        })
    except Exception as e:
        test_results.append({
            'endpoint': '/',
            'method': 'GET',
            'status_code': 0,
            'response_time': 0,
            'success': False,
            'error': str(e)
        })
    
    # Test health endpoint
    try:
        response = requests.get(f"{endpoint}/health")
        test_results.append({
            'endpoint': '/health',
            'method': 'GET',
            'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds(),
            'success': response.status_code == 200
        })
    except Exception as e:
        test_results.append({
            'endpoint': '/health',
            'method': 'GET',
            'status_code': 0,
            'response_time': 0,
            'success': False,
            'error': str(e)
        })
    
    # Test model info endpoint
    try:
        response = requests.get(f"{endpoint}/model/info")
        test_results.append({
            'endpoint': '/model/info',
            'method': 'GET',
            'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds(),
            'success': response.status_code == 200
        })
    except Exception as e:
        test_results.append({
            'endpoint': '/model/info',
            'method': 'GET',
            'status_code': 0,
            'response_time': 0,
            'success': False,
            'error': str(e)
        })
    
    return test_results

def test_single_predictions(endpoint, samples):
    """Test individual predictions"""
    print("🔮 Testing single predictions...")
    
    prediction_results = []
    
    for i, sample in enumerate(samples[:10]):  # Test first 10 samples
        try:
            start_time = time.time()
            response = requests.post(f"{endpoint}/predict", json=sample, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                prediction_results.append({
                    'sample_id': i + 1,
                    'input': sample,
                    'prediction': result['prediction'],
                    'probability': result['probability'],
                    'risk_level': result['risk_level'],
                    'confidence': result['confidence'],
                    'response_time': response_time,
                    'success': True
                })
                
                print(f"Sample {i+1}: {result['risk_level']} (confidence: {result['confidence']:.3f})")
            else:
                prediction_results.append({
                    'sample_id': i + 1,
                    'input': sample,
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'response_time': response_time
                })
                
        except Exception as e:
            prediction_results.append({
                'sample_id': i + 1,
                'input': sample,
                'success': False,
                'error': str(e),
                'response_time': 0
            })
    
    return prediction_results

def test_batch_predictions(endpoint, samples):
    """Test batch predictions"""
    print("📦 Testing batch predictions...")
    
    batch_size = 20
    batch_samples = samples[:batch_size]
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{endpoint}/predict/batch", 
            json={"samples": batch_samples}, 
            timeout=60
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return {
                'batch_size': batch_size,
                'predictions_count': len(result['predictions']),
                'response_time': response_time,
                'success': True,
                'predictions': result['predictions'][:5]  # Store only first 5 for space
            }
        else:
            return {
                'batch_size': batch_size,
                'success': False,
                'error': f"HTTP {response.status_code}",
                'response_time': response_time
            }
            
    except Exception as e:
        return {
            'batch_size': batch_size,
            'success': False,
            'error': str(e),
            'response_time': 0
        }

def test_metrics_endpoint(endpoint):
    """Test Prometheus metrics endpoint"""
    print("📊 Testing metrics endpoint...")
    
    try:
        response = requests.get(f"{endpoint}/metrics")
        if response.status_code == 200:
            metrics_data = response.text
            # Count different metric types
            lines = metrics_data.split('\n')
            metric_count = len([line for line in lines if line and not line.startswith('#')])
            
            return {
                'success': True,
                'status_code': response.status_code,
                'metrics_count': metric_count,
                'sample_metrics': lines[:10]  # First 10 lines
            }
        else:
            return {
                'success': False,
                'status_code': response.status_code
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def analyze_performance(prediction_results):
    """Analyze API performance"""
    print("📈 Analyzing performance...")
    
    successful_predictions = [r for r in prediction_results if r.get('success', False)]
    
    if not successful_predictions:
        return {'error': 'No successful predictions to analyze'}
    
    response_times = [r['response_time'] for r in successful_predictions]
    
    performance_stats = {
        'total_requests': len(prediction_results),
        'successful_requests': len(successful_predictions),
        'success_rate': len(successful_predictions) / len(prediction_results),
        'avg_response_time': np.mean(response_times),
        'median_response_time': np.median(response_times),
        'min_response_time': np.min(response_times),
        'max_response_time': np.max(response_times),
        'p95_response_time': np.percentile(response_times, 95),
        'p99_response_time': np.percentile(response_times, 99)
    }
    
    return performance_stats

def generate_observability_report(endpoint, test_results, prediction_results, 
                                batch_result, metrics_result, performance_stats):
    """Generate comprehensive observability report"""
    
    report = []
    report.append("🔍 API TESTING & OBSERVABILITY REPORT")
    report.append("="*50)
    report.append(f"Test Time: {datetime.now().isoformat()}")
    report.append(f"API Endpoint: {endpoint}")
    
    # Endpoint tests
    report.append("\n📡 ENDPOINT TESTS:")
    for test in test_results:
        status = "✅ PASS" if test['success'] else "❌ FAIL"
        report.append(f"  {test['endpoint']} ({test['method']}): {status}")
        report.append(f"    Status: {test['status_code']}, Response Time: {test['response_time']:.3f}s")
    
    # Prediction tests
    report.append("\n🔮 PREDICTION TESTS:")
    successful_preds = len([r for r in prediction_results if r.get('success', False)])
    report.append(f"  Single Predictions: {successful_preds}/{len(prediction_results)} successful")
    
    if batch_result:
        batch_status = "✅ PASS" if batch_result['success'] else "❌ FAIL"
        report.append(f"  Batch Predictions: {batch_status}")
        if batch_result['success']:
            report.append(f"    Batch Size: {batch_result['batch_size']}")
            report.append(f"    Response Time: {batch_result['response_time']:.3f}s")
    
    # Performance analysis
    report.append("\n📊 PERFORMANCE ANALYSIS:")
    if 'error' not in performance_stats:
        report.append(f"  Success Rate: {performance_stats['success_rate']:.1%}")
        report.append(f"  Avg Response Time: {performance_stats['avg_response_time']:.3f}s")
        report.append(f"  Median Response Time: {performance_stats['median_response_time']:.3f}s")
        report.append(f"  95th Percentile: {performance_stats['p95_response_time']:.3f}s")
        report.append(f"  99th Percentile: {performance_stats['p99_response_time']:.3f}s")
    
    # Metrics endpoint
    report.append("\n📈 METRICS ENDPOINT:")
    if metrics_result['success']:
        report.append(f"  ✅ Prometheus metrics available")
        report.append(f"  Metrics Count: {metrics_result['metrics_count']}")
    else:
        report.append(f"  ❌ Metrics endpoint failed")
    
    # Recommendations
    report.append("\n💡 RECOMMENDATIONS:")
    
    if performance_stats.get('success_rate', 0) < 0.95:
        report.append("  ⚠️ Success rate below 95% - investigate errors")
    
    if performance_stats.get('avg_response_time', 0) > 1.0:
        report.append("  ⚠️ Average response time > 1s - consider optimization")
    
    if performance_stats.get('p95_response_time', 0) > 2.0:
        report.append("  ⚠️ 95th percentile > 2s - check for latency spikes")
    
    if not metrics_result.get('success', False):
        report.append("  ⚠️ Set up monitoring with metrics endpoint")
    
    report.append("\n✅ API testing completed successfully!")
    
    return "\n".join(report)

def save_results(samples, test_results, prediction_results, batch_result, 
                metrics_result, performance_stats, report):
    """Save all testing results"""
    print("💾 Saving test results...")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_samples': samples,
        'endpoint_tests': test_results,
        'prediction_tests': prediction_results,
        'batch_test': batch_result,
        'metrics_test': metrics_result,
        'performance_stats': performance_stats,
        'report': report
    }
    
    with open('results/step4_api_testing/api_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save report as text file
    with open('results/step4_api_testing/observability_report.txt', 'w') as f:
        f.write(report)
    
    # Save samples as CSV for other steps
    df_samples = pd.DataFrame(samples)
    df_samples.to_csv('results/step4_api_testing/test_samples.csv', index=False)

def main():
    """Main execution function"""
    print("🚀 Starting Step 4: API Testing & Observability")
    print("="*50)
    
    create_directories()
    
    # Get service endpoint
    endpoint = get_service_endpoint()
    if not endpoint:
        print("❌ Could not determine service endpoint")
        return
    
    print(f"🔗 Testing endpoint: {endpoint}")
    
    # Wait for service to be ready
    if not wait_for_service(endpoint):
        print("❌ Service not available")
        return
    
    # Generate test samples
    samples = generate_random_samples(100)
    
    # Run tests
    test_results = test_api_endpoints(endpoint)
    prediction_results = test_single_predictions(endpoint, samples)
    batch_result = test_batch_predictions(endpoint, samples)
    metrics_result = test_metrics_endpoint(endpoint)
    
    # Analyze performance
    performance_stats = analyze_performance(prediction_results)
    
    # Generate report
    report = generate_observability_report(
        endpoint, test_results, prediction_results, batch_result, 
        metrics_result, performance_stats
    )
    
    print(report)
    
    # Save results
    save_results(samples, test_results, prediction_results, batch_result,
                metrics_result, performance_stats, report)
    
    print("\n✅ Step 4 completed successfully!")
    print("📊 Results saved to: results/step4_api_testing/")

if __name__ == "__main__":
    main()
