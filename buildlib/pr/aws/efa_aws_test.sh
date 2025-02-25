#!/bin/bash

set -exE
source ./buildlib/az-helpers.sh

# Generate properties json from template
envsubst < buildlib/pr/aws/efa_vars.template > efa_vars.json
jq '.' efa_vars.json

# Submit AWS batch job and capture job ID
aws eks update-kubeconfig --name ucx-ci
JOB_ID=$(aws batch submit-job \
    --job-name UCX_CI_"${BUILD_NUMBER}" \
    --job-definition UCX-CI-JD \
    --job-queue ucx-ci-JQ \
    --eks-properties-override file://./efa_vars.json \
    --tags JOB_URL="${JOB_URL}" \
    --query 'jobId' --output text)

# Set global variable in Azure
azure_set_variable "JOB_ID" "${JOB_ID}"

# Wait for job to start running
STATUS_CHECK="aws batch describe-jobs --jobs $JOB_ID --query 'jobs[0].status' --output text"
TIMEOUT=600
ELAPSED=0

while ! eval "$STATUS_CHECK" | grep -q RUNNING; do
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "Failed to start, status: $(eval "$STATUS_CHECK")"
        exit 1
    fi
    sleep 20
    ELAPSED=$((ELAPSED + 20))
done

# Get pod name and stream logs
POD=$(aws batch describe-jobs --jobs "$JOB_ID" --query jobs[0].eksProperties.podProperties.podName --output text)
kubectl -n ucx-ci-batch-nodes logs -f "$POD"

# Propagate exit status
if eval "$STATUS_CHECK" | grep -q FAILED; then  
  msg="Failure running EFA test in AWS"  
  azure_log_error "$msg"
  azure_complete_with_issues "$msg" 
fi
