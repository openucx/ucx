#!/bin/bash -eE

echo "Running $0 $*..."
eval "$*"
initial_delay=${initial_delay:=10}
interface=${interface:=bond0}
cycles=${cycles:=1000}
downtime=${downtime:=5}
uptime=${uptime:=20}
reset=${reset:="no"}


manager_script_dir=/hpc/noarch/git_projects/hpc-mtt-conf/scripts
manager_script=${manager_script_dir}/switch_port_on_off.py

if [ "x$reset" = "xyes" ]; then
    echo "Resetting interface ${interface} on $(hostname) interface ..."
    ${manager_script} -d ${manager_script_dir}/hosts --host $(hostname) -a on -i ${interface}
    sleep "$uptime"
    exit $?
fi

echo "Initial delay ${initial_delay} sec"
sleep ${initial_delay}

for i in $(seq 1 ${cycles}); do
    echo "#$i Put it down! And sleep ${downtime}"
    ${manager_script} --one -d ${manager_script_dir}/hosts --host $(hostname) -a off -i ${interface}
    sleep "$downtime"

    echo "#$i Put it up! And sleep ${uptime}"
    ${manager_script} -d ${manager_script_dir}/hosts --host $(hostname) -a on -i ${interface}
    sleep "$uptime"
done
