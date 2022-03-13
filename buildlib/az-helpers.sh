#!/bin/bash -eE

# The following functions uses Azure logging commands to report test
# details or errors. If the process is not running in Azure environment,
# no special output is generated.

# Logging commands documentation: https://docs.microsoft.com/en-us/azure/devops/pipelines/scripts/logging-commands


RUNNING_IN_AZURE="yes"
if [ -z "$AGENT_ID" ]; then
    RUNNING_IN_AZURE="no"
fi

# Report error and exit
function error() {
    msg=$1
    azure_log_issue "${msg}"
    echo "ERROR: ${msg}"
    exit 1
}

# Define Azure pipeline variable
function azure_set_variable() {
    test "x$RUNNING_IN_AZURE" = "xno" && return
    name=$1
    value=$2
    set +x
    echo "##vso[task.setvariable variable=${name}]${value}"
}

# Report an issue to Azure pipeline and stop step execution
function azure_log_issue() {
    test "x$RUNNING_IN_AZURE" = "xno" && return
    msg=$1
    set +x
    echo "##vso[task.logissue type=error]${msg}"
    echo "##vso[task.complete result=Failed;]"
}

# Report an error message to Azure pipeline
function azure_log_error() {
    test "x$RUNNING_IN_AZURE" = "xno" && return
    msg=$1
    set +x
    echo "##vso[task.logissue type=error]${msg}"
}

# Report an warning message to Azure pipeline
function azure_log_warning() {
    test "x$RUNNING_IN_AZURE" = "xno" && return
    msg=$1
    set +x
    echo "##vso[task.logissue type=warning]${msg}"
}

# Complete the task as "succeeeded with issues"
function azure_complete_with_issues() {
    test "x$RUNNING_IN_AZURE" = "xno" && return
    msg=$1
    set +x
    echo "##vso[task.complete result=SucceededWithIssues;]DONE${msg}"
}

# Get IPv4 address of an interface
function get_ip() {
    iface=$1
    ip=$(ip addr show "$iface" | awk '/inet / {print $2}' | awk -F/ '{print $1}')
    echo "$ip"
}

# Get active RDMA interfaces
function get_rdma_interfaces() {
    ibdev2netdev | grep Up | while read line
    do
        ibdev=$(echo "${line}" | awk '{print $1}')
        port=$(echo "${line}" | awk '{print $3}')
        netif=$(echo "${line}" | awk '{print $5}')

        # skip devices that do not have proper gid (representors)
        if ! [ -e "/sys/class/infiniband/${ibdev}/ports/${port}/gids/0" ]
        then
            continue
        fi

        echo ${netif}
    done | sort -u
}

# Prepend each line with a timestamp
function add_timestamp() {
    set +x
    while IFS= read -r line; do
        echo "$(date -u +"%Y-%m-%dT%T.%NZ") $line"
    done
}

function az_init_modules() {
    . /etc/profile.d/modules.sh
    export MODULEPATH="/hpc/local/etc/modulefiles:$MODULEPATH"
}

#
# Test if an environment module exists and load it if yes.
# Otherwise, return error code.
#
function az_module_load() {
    module=$1

    if module avail -t 2>&1 | grep -q "^$module\$"
    then
        module load $module
        return 0
    else
        echo "MODULEPATH='${MODULEPATH}'"
        module avail || true
        azure_log_warning "Module $module cannot be loaded"
        return 1
    fi
}

#
# Safe unload for env modules (even if it doesn't exist)
#
function az_module_unload() {
    module=$1
    module unload "${module}" || true
}


#
# try load cuda modules if nvidia driver is installed
#
try_load_cuda_env() {
    num_gpus=0
    have_cuda=no
    have_gdrcopy=no
    if [ -f "/proc/driver/nvidia/version" ]; then
        have_cuda=yes
        have_gdrcopy=yes
        az_module_load dev/cuda11.4 || have_cuda=no
        az_module_load dev/gdrcopy2.3_cuda11.4 || have_gdrcopy=no
        nvidia-smi -a
        ls -l /dev/nvidia*
        num_gpus=$(nvidia-smi -L | wc -l)
        if [ "$num_gpus" -gt 0 ] && ! [ -f /sys/kernel/mm/memory_peers/nv_mem/version ]
        then
            lsmod
            azure_log_error "GPU direct driver not loaded"
        fi
    fi
}


check_release_build() {
    build_reason=$1
    build_sourceversion=$2
    title_mask=$3


    if [ "${build_reason}" == "IndividualCI" ] || \
       [ "${build_reason}" == "ResourceTrigger" ]
    then
        launch=True
    elif [ "${build_reason}" == "PullRequest" ]
    then
        launch=False
        # In case of pull request, HEAD^ is the branch commit we merge with
        range="$(git rev-parse HEAD^)..${build_sourceversion}"
        for sha1 in `git log $range --format="%h"`
        do
            title=`git log -1 --format="%s" $sha1`
            [[ "$title" == "${title_mask}"* ]] && launch=True;
        done
    fi

    echo "##vso[task.setvariable variable=Launch;isOutput=true]${launch}"
}
