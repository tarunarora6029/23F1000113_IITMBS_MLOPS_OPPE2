#!/bin/bash

# Deployment script for Heart Disease API
set -e

IMAGE_URL=${1:-"gcr.io/$PROJECT_ID/heart-disease-model:latest"}

echo "🚀 Deploying Heart Disease API to Kubernetes"
echo "Image: $IMAGE_URL"
echo "==========================================="

# Replace image URL in deployment
export IMAGE_URL=$IMAGE_URL
envsubst < k8s/deployment.yaml | kubectl apply -f -

# Apply other K8s resources
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Wait for deployment to complete
echo "⏳ Waiting for deployment to complete..."
kubectl rollout status deployment/heart-disease-api --timeout=300s

# Get service information
echo "📊 Service Information:"
kubectl get services heart-disease-service

# Get pod information
echo "🔍 Pod Status:"
kubectl get pods -l app=heart-disease-api

# Wait for external IP
echo "⏳ Waiting for external IP..."
for i in {1..30}; do
    EXTERNAL_IP=$(kubectl get svc heart-disease-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [[ -n "$EXTERNAL_IP" ]]; then
        echo "✅ Service available at: http://$EXTERNAL_IP"
        break
    fi
    echo "Waiting for external IP... (attempt $i/30)"
    sleep 10
done

if [[ -z "$EXTERNAL_IP" ]]; then
    echo "⚠️ External IP not assigned yet. Check service status manually."
else
    echo "🎉 Deployment completed successfully!"
    echo "API endpoint: http://$EXTERNAL_IP"
    echo "Health check: http://$EXTERNAL_IP/health"
    echo "API docs: http://$EXTERNAL_IP/docs"
fi
