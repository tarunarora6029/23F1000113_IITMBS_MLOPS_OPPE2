#!/bin/bash

# Deployment script for Heart Disease API
set -e

if [[ -z "$1" ]]; then
  echo "❌ Usage: $0 <IMAGE_URL>"
  exit 1
fi

IMAGE_URL=$1
LATEST_IMAGE=$(echo $IMAGE_URL | sed -E 's/:.+$/:latest/')

echo "🚀 Deploying Heart Disease API to Kubernetes"
echo "Using Image (SHA pinned): $IMAGE_URL"
echo "Also tagging as:          $LATEST_IMAGE"
echo "==========================================="

# Tag & push SHA image as "latest" too
docker pull $IMAGE_URL || true   # ensure local availability
docker tag $IMAGE_URL $LATEST_IMAGE
docker push $LATEST_IMAGE

# Replace image URL in deployment (SHA pinned)
export IMAGE_URL=$IMAGE_URL
envsubst < k8s/deployment.yaml | kubectl apply -f -

# Apply other K8s resources
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Wait for deployment to complete (bump timeout)
echo "⏳ Waiting for deployment to complete..."
if ! kubectl rollout status deployment/heart-disease-api --timeout=600s; then
  echo "❌ Rollout timed out. Collecting diagnostics…"
  kubectl get pods -l app=heart-disease-api -o wide
  echo "---- Events (namespace default) ----"
  kubectl get events --sort-by=.lastTimestamp | tail -n 100
  echo "---- Describe Pods ----"
  for p in $(kubectl get pods -l app=heart-disease-api -o name); do
    echo "==== $p ===="; kubectl describe $p; echo;
  done
  echo "---- Recent Logs ----"
  for p in $(kubectl get pods -l app=heart-disease-api -o name); do
    echo "==== $p ===="; kubectl logs --tail=200 $p || true; echo;
  done
  exit 1
fi

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
