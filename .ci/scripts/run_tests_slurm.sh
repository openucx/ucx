#!/bin/bash -xe
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.

set -o pipefail

function usage()
{
cat <<EOF
Usage: $0 <[options=value]>
Options:
--test_script_path            Path to the test script
--docker_image                Docker image name
--slurm_job_id                SLURM job ID
--slurm_nodes                 Number of SLURM nodes
--slurm_head_node             SLURM head node (optional, uses SLURM_HEAD_NODE env if not set)
--slurm_head_user             SSH user for SLURM head node (optional, used with dlcluster)
--slurm_env                   Comma-separated list of VAR=val pairs to export into the container (optional)
--container_name              Container name (optional, uses "ucx-\${BUILD_NUMBER}" if not set)
EOF
exit 1
}

[ $# -eq 0 ] && usage
while getopts ":h-:" optchar; do
    case "${optchar}" in
        -)
            case "${OPTARG}" in
                test_script_path=*)
                    test_script_path=${OPTARG#*=}
                    ;;
                docker_image=*)
                    docker_image=${OPTARG#*=}
                    ;;
                slurm_job_id=*)
                    slurm_job_id=${OPTARG#*=}
                    ;;
                slurm_nodes=*)
                    slurm_nodes=${OPTARG#*=}
                    ;;
                slurm_head_node=*)
                    slurm_head_node=${OPTARG#*=}
                    ;;
                slurm_head_user=*)
                    slurm_head_user=${OPTARG#*=}
                    ;;
                slurm_env=*)
                    slurm_env=${OPTARG#*=}
                    ;;
                container_name=*)
                    container_name=${OPTARG#*=}
                    ;;
                *)
                    echo "Unknown option --${OPTARG}" >&2
                    exit 1
                    ;;
            esac;;
        h | *)
            usage
            exit 0
            ;;
    esac
done

# Use environment variables as fallback
docker_image=${docker_image:-${DOCKER_IMAGE_NAME}}
slurm_job_id=${slurm_job_id:-${SLURM_JOB_ID}}
slurm_nodes=${slurm_nodes:-${SLURM_NODES}}
slurm_head_node=${slurm_head_node:-${SLURM_HEAD_NODE}}
slurm_head_user=${slurm_head_user:-${SLURM_HEAD_USER}}
container_name=${container_name:-"ucx-${BUILD_NUMBER}"}

# Validate required parameters
: ${docker_image:?Missing --docker_image}
: ${test_script_path:?Missing --test_script_path}

# Build SLURM command using bash arrays
SLURM_CMD=(
    "srun"
    "--jobid=${slurm_job_id}"
    "--nodes=${slurm_nodes}"
    "--mpi=pmix"
    "--container-writable"
    "--container-name=${container_name}"
    "--container-image=${docker_image}"
)

# Add optional environment variable exports
[ -n "${slurm_env}" ] && SLURM_CMD+=("--export=${slurm_env}")

SLURM_CMD+=(bash -c "${test_script_path}")

echo "INFO: Executing test script: ${test_script_path}"
echo "INFO: Using SLURM job ID: ${slurm_job_id}"
echo "INFO: Using Docker image: ${docker_image}"
echo "INFO: Container name: ${container_name}"
echo "INFO: SLURM command: ${SLURM_CMD[*]}"

# Validate SLURM_HEAD_NODE is set
if [ -z "${slurm_head_node}" ]; then
    echo "ERROR: SLURM_HEAD_NODE is not set or empty"
    exit 1
fi

# Execute based on head node type
case "${slurm_head_node}" in
    scctl)
        echo "INFO: Using scctl client to connect and execute SLURM command"
        scctl --raw-errors client connect -- "$(printf '%q ' "${SLURM_CMD[@]}")"
        ;;
    dlcluster*)
        echo "INFO: Using SSH to connect to ${slurm_head_node} and execute SLURM command"
        # Construct SSH target with optional user
        ssh_target="${slurm_head_node}"
        [ -n "${slurm_head_user}" ] && ssh_target="${slurm_head_user}@${slurm_head_node}"
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${ssh_target}" "$(printf '%q ' "${SLURM_CMD[@]}")"
        ;;
    *)
        echo "ERROR: Invalid SLURM_HEAD_NODE value: ${slurm_head_node}"
        echo "Supported values: scctl, dlcluster, dlcluster.nvidia.com"
        exit 1
        ;;
esac

echo "INFO: Test execution completed successfully"
