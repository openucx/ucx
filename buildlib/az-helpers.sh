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

# Prepend each line with a timestamp
function add_timestamp() {
    set +x
    while IFS= read -r line; do
        echo "$(date -u +"%Y-%m-%dT%T.%NZ") $line"
    done
}
