#!/bin/bash -eEx

# Search for zombie or defunct processes; kill them all
out=$(ps -eo pid,stat | awk '$2 == "Z|defunct" {print $1}')
if [ -z "$out" ]; then
  echo "No zombies found"
  exit 0
else
  for i in $out ; do
    kill -9 "$i"
  done
fi

# Search again; tell parents you killed their kids; kill parents
out=$(ps -eo pid,stat | awk '$2 == "Z|defunct" {print $1}')
for i in $out ; do
  pid_parent=$(ps -o ppid=$i)
  kill -s SIGCHLD $pid_parent
  kill -9 $pid_parent
done
