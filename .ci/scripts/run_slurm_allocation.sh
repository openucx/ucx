#!/bin/bash
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
set -xe

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

function usage()
{
cat <<EOF
Usage: $0 <[options=value]>
Options:
--slurm_job_name              SLURM job name
--slurm_partition             SLURM partition
--slurm_nodes                 Number of SLURM nodes
--slurm_node_name             Specific node name to use (optional)
--slurm_head_node             SLURM head node
--slurm_head_user             SSH user for SLURM head node (optional, used with dlcluster)
--slurm_job_timeout           SLURM job timeout
--slurm_job_id_file           File path to save the SLURM job ID
--slurm_account               SLURM account (optional)
--slurm_gres                  SLURM GRES specification (optional)
--slurm_mem                   SLURM memory allocation (optional)
--slurm_mincpus               SLURM minimum CPUs (optional)
--slurm_exclude               SLURM nodes to exclude (optional)
--workspace                   Workspace directory
EOF
exit 1
}

while getopts ":h-:" optchar; do
    case "${optchar}" in
        -)
            case "${OPTARG}" in
                slurm_job_name=*)
                    slurm_job_name=${OPTARG#*=}
                    ;;
                slurm_partition=*)
                    slurm_partition=${OPTARG#*=}
                    ;;
                slurm_nodes=*)
                    slurm_nodes=${OPTARG#*=}
                    ;;
                slurm_node_name=*)
                    slurm_node_name=${OPTARG#*=}
                    ;;
                slurm_head_node=*)
                    slurm_head_node=${OPTARG#*=}
                    ;;
                slurm_head_user=*)
                    slurm_head_user=${OPTARG#*=}
                    ;;
                slurm_job_timeout=*)
                    slurm_job_timeout=${OPTARG#*=}
                    ;;
                slurm_job_id_file=*)
                    slurm_job_id_file=${OPTARG#*=}
                    ;;
                slurm_account=*)
                    slurm_account=${OPTARG#*=}
                    ;;
                slurm_gres=*)
                    slurm_gres=${OPTARG#*=}
                    ;;
                slurm_mem=*)
                    slurm_mem=${OPTARG#*=}
                    ;;
                slurm_mincpus=*)
                    slurm_mincpus=${OPTARG#*=}
                    ;;
                slurm_exclude=*)
                    slurm_exclude=${OPTARG#*=}
                    ;;
                workspace=*)
                    workspace=${OPTARG#*=}
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

slurm_job_name=${slurm_job_name:-${SLURM_JOB_NAME}}
slurm_partition=${slurm_partition:-${SLURM_PARTITION}}
slurm_nodes=${slurm_nodes:-${SLURM_NODES}}
slurm_node_name=${slurm_node_name:-${SLURM_NODE_NAME}}
slurm_head_node=${slurm_head_node:-${SLURM_HEAD_NODE}}
slurm_head_user=${slurm_head_user:-${SLURM_HEAD_USER}}
slurm_job_timeout=${slurm_job_timeout:-${SLURM_JOB_TIMEOUT}}
slurm_job_id_file=${slurm_job_id_file:-${SLURM_JOB_ID_FILE}}
slurm_account=${slurm_account:-${SLURM_ACCOUNT}}
slurm_gres=${slurm_gres:-${SLURM_GRES}}
slurm_mem=${slurm_mem:-${SLURM_MEM}}
slurm_mincpus=${slurm_mincpus:-${SLURM_MINCPUS}}
slurm_exclude=${slurm_exclude:-${SLURM_EXCLUDE}}
workspace=${workspace:-${WORKSPACE}}

# Set default job ID file path if not specified
if [ -z "${slurm_job_id_file}" ] && [ -n "${workspace}" ]; then
    slurm_job_id_file="${workspace}/job_id.txt"
fi

: ${slurm_job_name:?Missing --slurm_job_name}
: ${slurm_partition:?Missing --slurm_partition}
: ${slurm_nodes:?Missing --slurm_nodes}
: ${slurm_head_node:?Missing --slurm_head_node}
: ${slurm_job_timeout:?Missing --slurm_job_timeout}

readonly SLURM_IMMEDIATE_TIMEOUT=${SLURM_IMMEDIATE_TIMEOUT:-600} # time to wait for resource allocation to be granted

# Build SLURM allocation command
SLURM_ALLOC_ARGS=(
    "salloc"
)

# Add optional account specification
[ -n "${slurm_account}" ] && SLURM_ALLOC_ARGS+=("--account=${slurm_account}")

# Add required allocation parameters
SLURM_ALLOC_ARGS+=(
    "-N" "${slurm_nodes}"
    "-p" "${slurm_partition}"
)

# Add optional resource specifications
[ -n "${slurm_gres}" ] && SLURM_ALLOC_ARGS+=("--gres=${slurm_gres}")
[ -n "${slurm_mem}" ] && SLURM_ALLOC_ARGS+=("--mem=${slurm_mem}")
[ -n "${slurm_mincpus}" ] && SLURM_ALLOC_ARGS+=("--mincpus=${slurm_mincpus}")
[ -n "${slurm_node_name}" ] && SLURM_ALLOC_ARGS+=("--nodelist=${slurm_node_name}")
[ -n "${slurm_exclude}" ] && SLURM_ALLOC_ARGS+=("--exclude=${slurm_exclude}")

# Add required job parameters
SLURM_ALLOC_ARGS+=(
    "--job-name=${slurm_job_name}"
    "--immediate=${SLURM_IMMEDIATE_TIMEOUT}"
    "--time=${slurm_job_timeout}"
    "--no-shell"
)

readonly SLURM_ALLOCATION_CMD="${SLURM_ALLOC_ARGS[*]}"
readonly SLURM_GET_JOB_ID_CMD="squeue --noheader --name=${slurm_job_name} --format=%i"

case "${slurm_head_node}" in
    scctl)
        echo "INFO: Using scctl client to connect and allocate Slurm resources"
        export SCCTL_USER=${SERVICE_USER_USERNAME}
        export SCCTL_PASSWORD=${SERVICE_USER_PASSWORD}
        scctl -v
        scctl --raw-errors upgrade
        scctl --raw-errors login
        result=$(scctl --raw-errors client exists)
        if [ "$result" == "client does not exist" ]; then
            echo "INFO: Creating scctl client"
            scctl --raw-errors client create
        fi
        echo "INFO: Allocating Slurm resources via scctl"
        scctl --raw-errors client connect -- "${SLURM_ALLOCATION_CMD}"
        JOB_ID=$(scctl --raw-errors client connect -- "${SLURM_GET_JOB_ID_CMD}")
        ;;
    dlcluster*)
        echo "INFO: Using SSH to connect to ${slurm_head_node} and allocate Slurm resources"

        # Construct SSH target with optional user
        ssh_target="${slurm_head_node}"
        [ -n "${slurm_head_user}" ] && ssh_target="${slurm_head_user}@${slurm_head_node}"

        allocation_output=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${ssh_target}" "${SLURM_ALLOCATION_CMD}" 2>&1)
        echo "${allocation_output}"

        # Extract job ID from salloc output (format: "Granted job allocation 123456")
        JOB_ID=$(echo "${allocation_output}" | grep -oP "Granted job allocation \K[0-9]+")

        if [ -z "${JOB_ID}" ]; then
            echo "ERROR: Failed to extract job ID from allocation output"
            echo "Allocation output: ${allocation_output}"
            exit 1
        fi
        ;;
    *)
        echo "ERROR: Invalid SLURM_HEAD_NODE value: ${slurm_head_node}"
        echo "Supported values: scctl, dlcluster, dlcluster.nvidia.com"
        exit 1
        ;;
esac

: ${JOB_ID:?Failed to get job ID}

echo "INFO: Job ID: ${JOB_ID}"
if [ -n "${slurm_job_id_file}" ]; then
    echo "${JOB_ID}" > "${slurm_job_id_file}"
    echo "INFO: Job ID saved to ${slurm_job_id_file}"
fi
echo "INFO: Slurm resources allocated successfully"
