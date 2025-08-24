#!/bin/bash

# Step 5: Performance Testing with wrk
# High concurrency load testing and timeout analysis

set -e

echo "🚀 Starting Step 5: Performance Testing with wrk"
echo "==============================================="

# Create results directory
mkdir -p results/step5_performance

# Get service endpoint
echo "🔗 Getting service endpoint..."
EXTERNAL_IP=$(kubectl get service heart-disease-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

if [[ -z "$EXTERNAL_IP" ]]; then
    echo "⚠️ No external IP found, trying NodePort..."
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}')
    NODE_PORT=$(kubectl get service heart-disease-service -o jsonpath='{.spec.ports[0].nodePort}')
    ENDPOINT="http://$NODE_IP:$NODE_PORT"
else
    ENDPOINT="http://$EXTERNAL_IP"
fi

echo "📡 Testing endpoint: $ENDPOINT"

# Verify service is running
echo "🔍 Verifying service availability..."
if ! curl -f "$ENDPOINT/health" &>/dev/null; then
    echo "❌ Service is not responding"
    exit 1
fi

echo "✅ Service is available"

# Create test payload
cat > /tmp/test_payload.json << 'EOF'
{
    "age": 55,
    "gender": 1,
    "cp": 2,
    "trestbps": 140.0,
    "chol": 250.0,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150.0,
    "exang": 1,
    "oldpeak": 2.5,
    "slope": 1,
    "ca": 1,
    "thal": 2
}
EOF

# Create wrk Lua script for POST requests
cat > /tmp/post_script.lua << 'EOF'
wrk.method = "POST"
wrk.body = '{"age":55,"gender":1,"cp":2,"trestbps":140.0,"chol":250.0,"fbs":0,"restecg":1,"thalach":150.0,"exang":1,"oldpeak":2.5,"slope":1,"ca":1,"thal":2}'
wrk.headers["Content-Type"] = "application/json"
EOF

echo ""
echo "🧪 PERFORMANCE TESTING SCENARIOS"
echo "================================"

# Test 1: Baseline - Low concurrency
echo ""
echo "📊 Test 1: Baseline Performance (2 threads, 10 connections, 30s)"
echo "----------------------------------------------------------------"
wrk -t2 -c10 -d30s --timeout=10s \
    --script=/tmp/post_script.lua \
    "$ENDPOINT/predict" \
    > results/step5_performance/test1_baseline.txt 2>&1

cat results/step5_performance/test1_baseline.txt

# Test 2: Moderate concurrency
echo ""
echo "📊 Test 2: Moderate Load (4 threads, 50 connections, 60s)"
echo "---------------------------------------------------------"
wrk -t4 -c50 -d60s --timeout=10s \
    --script=/tmp/post_script.lua \
    "$ENDPOINT/predict" \
    > results/step5_performance/test2_moderate.txt 2>&1

cat results/step5_performance/test2_moderate.txt

# Test 3: High concurrency
echo ""
echo "📊 Test 3: High Load (8 threads, 100 connections, 60s)"
echo "------------------------------------------------------"
wrk -t8 -c100 -d60s --timeout=15s \
    --script=/tmp/post_script.lua \
    "$ENDPOINT/predict" \
    > results/step5_performance/test3_high_load.txt 2>&1

cat results/step5_performance/test3_high_load.txt

# Test 4: Stress test - Very high concurrency
echo ""
echo "📊 Test 4: Stress Test (12 threads, 200 connections, 60s)"
echo "---------------------------------------------------------"
wrk -t12 -c200 -d60s --timeout=20s \
    --script=/tmp/post_script.lua \
    "$ENDPOINT/predict" \
    > results/step5_performance/test4_stress.txt 2>&1

cat results/step5_performance/test4_stress.txt

# Test 5: Burst test - Short high intensity
echo ""
echo "📊 Test 5: Burst Test (16 threads, 300 connections, 30s)"
echo "--------------------------------------------------------"
wrk -t16 -c300 -d30s --timeout=30s \
    --script=/tmp/post_script.lua \
    "$ENDPOINT/predict" \
    > results/step5_performance/test5_burst.txt 2>&1

cat results/step5_performance/test5_burst.txt

# Test 6: Batch endpoint performance
echo ""
echo "📊 Test 6: Batch Endpoint Test (4 threads, 20 connections, 30s)"
echo "----------------------------------------------------------------"

# Create batch payload
cat > /tmp/batch_payload.json << 'EOF'
{
    "samples": [
        {"age":55,"gender":1,"cp":2,"trestbps":140.0,"chol":250.0,"fbs":0,"restecg":1,"thalach":150.0,"exang":1,"oldpeak":2.5,"slope":1,"ca":1,"thal":2},
        {"age":45,"gender":0,"cp":1,"trestbps":120.0,"chol":200.0,"fbs":1,"restecg":0,"thalach":180.0,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":3},
        {"age":65,"gender":1,"cp":3,"trestbps":160.0,"chol":300.0,"fbs":0,"restecg":2,"thalach":130.0,"exang":1,"oldpeak":3.0,"slope":0,"ca":2,"thal":1},
        {"age":50,"gender":0,"cp":0,"trestbps":130.0,"chol":220.0,"fbs":1,"restecg":1,"thalach":170.0,"exang":0,"oldpeak":1.5,"slope":1,"ca":1,"thal":2},
        {"age":60,"gender":1,"cp":2,"trestbps":150.0,"chol":280.0,"fbs":0,"restecg":0,"thalach":140.0,"exang":1,"oldpeak":2.0,"slope":2,"ca":1,"thal":3}
    ]
}
EOF

# Create batch script
cat > /tmp/batch_script.lua << 'EOF'
wrk.method = "POST"
wrk.body = '{"samples":[{"age":55,"gender":1,"cp":2,"trestbps":140.0,"chol":250.0,"fbs":0,"restecg":1,"thalach":150.0,"exang":1,"oldpeak":2.5,"slope":1,"ca":1,"thal":2},{"age":45,"gender":0,"cp":1,"trestbps":120.0,"chol":200.0,"fbs":1,"restecg":0,"thalach":180.0,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":3},{"age":65,"gender":1,"cp":3,"trestbps":160.0,"chol":300.0,"fbs":0,"restecg":2,"thalach":130.0,"exang":1,"oldpeak":3.0,"slope":0,"ca":2,"thal":1},{"age":50,"gender":0,"cp":0,"trestbps":130.0,"chol":220.0,"fbs":1,"restecg":1,"thalach":170.0,"exang":0,"oldpeak":1.5,"slope":1,"ca":1,"thal":2},{"age":60,"gender":1,"cp":2,"trestbps":150.0,"chol":280.0,"fbs":0,"restecg":0,"thalach":140.0,"exang":1,"oldpeak":2.0,"slope":2,"ca":1,"thal":3}]}'
wrk.headers["Content-Type"] = "application/json"
EOF

wrk -t4 -c20 -d30s --timeout=30s \
    --script=/tmp/batch_script.lua \
    "$ENDPOINT/predict/batch" \
    > results/step5_performance/test6_batch.txt 2>&1

cat results/step5_performance/test6_batch.txt

# Monitor Kubernetes resources during testing
echo ""
echo "📊 Kubernetes Resources After Testing"
echo "====================================="
kubectl top pods -l app=heart-disease-api > results/step5_performance/pod_resources.txt 2>/dev/null || echo "Metrics server not available"
kubectl get hpa heart-disease-hpa > results/step5_performance/hpa_status.txt 2>/dev/null || echo "HPA not found"

# Parse and analyze results
echo ""
echo "🔍 PERFORMANCE ANALYSIS"
echo "======================="

python3 << 'EOF'
import re
import os
import json
from datetime import datetime

def parse_wrk_output(filename):
    """Parse wrk output file"""
    if not os.path.exists(filename):
        return None
        
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract key metrics using regex
    latency_avg = re.search(r'Latency\s+(\d+\.?\d*\w*)', content)
    latency_stdev = re.search(r'Stdev\s+(\d+\.?\d*\w*)', content)
    latency_max = re.search(r'Max\s+(\d+\.?\d*\w*)', content)
    
    req_sec = re.search(r'Req/Sec\s+(\d+\.?\d*)', content)
    
    total_requests = re.search(r'(\d+) requests in', content)
    total_time = re.search(r'requests in (\d+\.?\d*\w*)', content)
    
    errors = re.search(r'Non-2xx or 3xx responses: (\d+)', content)
    timeouts = re.search(r'Socket errors.*timeout (\d+)', content)
    
    return {
        'latency_avg': latency_avg.group(1) if latency_avg else 'N/A',
        'latency_stdev': latency_stdev.group(1) if latency_stdev else 'N/A',
        'latency_max': latency_max.group(1) if latency_max else 'N/A',
        'req_per_sec': float(req_sec.group(1)) if req_sec else 0,
        'total_requests': int(total_requests.group(1)) if total_requests else 0,
        'total_time': total_time.group(1) if total_time else 'N/A',
        'errors': int(errors.group(1)) if errors else 0,
        'timeouts': int(timeouts.group(1)) if timeouts else 0
    }

# Analyze all test results
tests = [
    ('test1_baseline.txt', 'Baseline (2t/10c)'),
    ('test2_moderate.txt', 'Moderate (4t/50c)'),
    ('test3_high_load.txt', 'High Load (8t/100c)'),
    ('test4_stress.txt', 'Stress (12t/200c)'),
    ('test5_burst.txt', 'Burst (16t/300c)'),
    ('test6_batch.txt', 'Batch Endpoint (4t/20c)')
]

results = {}
analysis = []

analysis.append("📊 PERFORMANCE SUMMARY")
analysis.append("=" * 50)
analysis.append(f"Test Time: {datetime.now().isoformat()}")
analysis.append("")

for filename, test_name in tests:
    filepath = f'results/step5_performance/{filename}'
    data = parse_wrk_output(filepath)
    
    if data:
        results[test_name] = data
        
        analysis.append(f"🧪 {test_name}:")
        analysis.append(f"   Requests/sec: {data['req_per_sec']:.1f}")
        analysis.append(f"   Avg Latency:  {data['latency_avg']}")
        analysis.append(f"   Max Latency:  {data['latency_max']}")
        analysis.append(f"   Total Reqs:   {data['total_requests']}")
        analysis.append(f"   Errors:       {data['errors']}")
        analysis.append(f"   Timeouts:     {data['timeouts']}")
        analysis.append("")

# Performance insights
analysis.append("💡 PERFORMANCE INSIGHTS:")

baseline_rps = results.get('Baseline (2t/10c)', {}).get('req_per_sec', 0)
high_load_rps = results.get('High Load (8t/100c)', {}).get('req_per_sec', 0)

if baseline_rps > 0 and high_load_rps > 0:
    throughput_ratio = high_load_rps / baseline_rps
    analysis.append(f"   Throughput scaling: {throughput_ratio:.2f}x from baseline to high load")

# Check for performance degradation
high_errors = sum(data.get('errors', 0) for data in results.values())
total_timeouts = sum(data.get('timeouts', 0) for data in results.values())

if high_errors > 0:
    analysis.append(f"   ⚠️  Total errors across all tests: {high_errors}")
if total_timeouts > 0:
    analysis.append(f"   ⚠️  Total timeouts across all tests: {total_timeouts}")

analysis.append("")
analysis.append("🎯 RECOMMENDATIONS:")

if baseline_rps < 100:
    analysis.append("   • Consider optimizing model inference time")
if high_load_rps < baseline_rps * 0.8:
    analysis.append("   • Performance degrades significantly under load")
if total_timeouts > 0:
    analysis.append("   • Increase timeout values or optimize response time")
if high_errors > 0:
    analysis.append("   • Investigate error causes and add error handling")

# Save analysis
analysis_text = '\n'.join(analysis)
print(analysis_text)

with open('results/step5_performance/performance_analysis.txt', 'w') as f:
    f.write(analysis_text)

# Save JSON results
with open('results/step5_performance/performance_results.json', 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'test_results': results,
        'analysis': analysis_text
    }, f, indent=2)
EOF

# Cleanup
rm -f /tmp/test_payload.json /tmp/batch_payload.json /tmp/post_script.lua /tmp/batch_script.lua

echo ""
echo "✅ Step 5: Performance testing completed!"
echo "📊 Results saved to: results/step5_performance/"
echo "📈 Key files:"
echo "   - performance_analysis.txt (Summary)"
echo "   - performance_results.json (Detailed results)"
echo "   - test*.txt (Individual test outputs)"
