# EFA Tests in AWS

Azure DevOps pipeline job for running UCX tests on AWS Elastic Fabric Adapter (EFA) environment.

## Overview

The pipeline job executes UCX tests in a real EFA environment on AWS infrastructure. It leverages AWS Batch to manage the compute resources and job execution.

## Prerequisites

- AWS credentials (stored as secret vars)
- Docker container with AWS tools (`aws_tools`)
- AWS resources are pre-configured using a separate project (internal link):   
https://gitlab-master.nvidia.com/nbu-swx/infrastructure/swx-infrastructure/-/tree/master/aws


## How It Works

1. Generates AWS Batch job config from `efa_vars.template` using Azure pipeline variables
2. Submits a job to AWS Batch using predefined job definition
3. Monitors job execution and streams logs from the EKS pod
4. Handles job cleanup and status reporting
