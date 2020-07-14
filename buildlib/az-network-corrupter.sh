#!/bin/bash -eE

echo "Running $0 $*..."
eval "$*"
initial_delay=${initial_delay:=10}
cycles=${cycles:=1000}
downtime=${downtime:=5}
uptime=${uptime:=20}
reset=${reset:="no"}


manager_script=/hpc/noarch/git_projects/swx_infrastructure/clusters/bin/manage_host_ports.sh

if [ "x$reset" = "xyes" ]; then
    echo "Resetting interface on $(hostname)..."
    sudo "$manager_script" "$(hostname)" "bond-up"
    exit $?
fi

echo "Initial delay ${initial_delay} sec"
sleep ${initial_delay}

for i in $(seq 1 ${cycles}); do
    echo "#$i Put it down! And sleep ${downtime}"
    sudo "$manager_script" "$(hostname)" "bond-down"
    sleep "$downtime"

    echo "#$i Put it up! And sleep ${uptime}"
    sudo "$manager_script" "$(hostname)" "bond-up"
    sleep "$uptime"
done
