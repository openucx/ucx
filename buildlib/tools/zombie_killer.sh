#!/bin/bash -eE

# Search for zombies; kill them all
out=$(ps -eo pid,stat,%cpu | grep Z | awk '{print $1}')
if [ -z "$out" ]; then
  echo "No zombies were found"
  exit 0
else
  for i in $out ; do
    kill -9 "$i"
  done
fi

# Search for survivors; tell parents you killed their kids; kill parents
out=$(ps -eo pid,stat,%cpu | grep Z | awk '$3>85.0 {print $1}')
for i in $out ; do
  pid_parent=$(ps -p "$i" -o ppid=)
  kill -s SIGCHLD $pid_parent
  kill -9 $pid_parent
done
