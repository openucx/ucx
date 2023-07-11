#!/bin/bash -eE

log_info() {
  proc_name=$(ps -p "$2" -o comm --no-headers)
  logger -p user.info -t zombie_killer "$1: PID $2, P_NAME $proc_name"
}

# Search for zombies; kill them all
out=$(ps -eo pid,stat,%cpu | grep Z | awk '{print $1}')
if [ -z "$out" ]; then
  echo "No zombies were found"
  exit 0
else
  for i in $out ; do
    log_info "Kill zombie" "$i"
    kill -9 "$i"
  done
fi

# Search for survivors; tell parents you killed their kids; kill parents
out=$(ps -eo pid,stat,%cpu | grep Z | awk '$3>85.0 {print $1}')
for i in $out ; do
  pid_parent=$(ps -p "$i" -o ppid=)
  log_info "Kill parent" "$pid_parent"
  kill -s SIGCHLD $pid_parent
  kill -9 $pid_parent
done
