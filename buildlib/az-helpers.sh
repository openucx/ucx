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
    echo `ibdev2netdev | grep Up | awk '{print $5}'`
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
		az_module_load dev/cuda11.1.1 || have_cuda=no
		az_module_load dev/gdrcopy2.1_cuda11.1.1 || have_gdrcopy=no
		num_gpus=$(nvidia-smi -L | wc -l)
	fi
}


check_commit_message() {
    git_id=$1
    title_mask=$2
    build_reason=$3
    echo "Get commit message target $git_id"
    title=`git log -1 --format="%s" $git_id`

    if [[ ( "$build_reason" == "IndividualCI" ) || ( "$title" == "$title_mask"* && "$build_reason" == "PullRequest" ) ]]
    then
        echo "##vso[task.setvariable variable=Launch;isOutput=true]Yes"
    else
        echo "##vso[task.setvariable variable=Launch;isOutput=true]No"
    fi
}
