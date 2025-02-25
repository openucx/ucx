#!/bin/bash -exE

source ./buildlib/az-helpers.sh

# Generate properties json from template
envsubst <buildlib/pr/aws/efa_vars.template >efa_vars.json
jq '.' efa_vars.json

# Submit AWS batch job and capture job ID
aws eks update-kubeconfig --name ucx-ci
JOB_ID=$(aws batch submit-job \
    --job-name UCX_CI_"${BUILD_NUMBER}" \
    --job-definition UCX-CI-JD \
    --job-queue ucx-ci-JQ \
    --eks-properties-override file://./efa_vars.json \
    --tags PR_ID="${PR_ID},\
        Azure_Pipeline_Name=${PIPE_NAME},\
        Azure_Pipeline_Stage=${PIPE_STAGE}" \
    --query 'jobId' --output text)

# Set global variable in Azure
set +x
echo "##vso[task.setvariable variable=JOB_ID;isOutput=true]${JOB_ID}"
set -x

wait_for_status() {
    local target_status=$1
    local timeout=600
    local interval=30
    local elapsed=0
    local status=""

    while [ $elapsed -lt $timeout ]; do
        status=$(aws batch describe-jobs --jobs "$JOB_ID" --query 'jobs[0].status' --output text)
        if echo "$status" | grep -qE "$target_status"; then
            echo "$status"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    echo "Timeout waiting for status $target_status. Final status: $status"
    return 1
}

# Wait for the job to start running
if ! wait_for_status "RUNNING"; then
    echo "Job failed to start"
    exit 1
fi

# Get pod name and stream logs
POD=$(aws batch describe-jobs --jobs "$JOB_ID" --query jobs[0].eksProperties.podProperties.podName --output text)
kubectl -n ucx-ci-batch-nodes logs -f "$POD"

# Propagate exit status
exit_status=$(wait_for_status "SUCCEEDED|FAILED")
if [[ "$exit_status" == "FAILED" ]]; then
    msg="Failure running EFA test in AWS"
    azure_log_error "$msg"
    azure_complete_with_issues "$msg"
fi
