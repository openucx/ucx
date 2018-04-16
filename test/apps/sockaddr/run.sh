#!/bin/sh
#
# This script is indended to run from a Slurm allocation.
# For example:run.sh -m tcp
#   $ salloc ...
#   $ ./build-devel/test/apps/sockaddr/run.sh -m tcp
#

mydir=$(dirname $0)
netdev=eth0
tcpport=30000
mode=tcp
wait_time=3
pdsh_args=""
hostfile=$mydir/sa_hostfile.$$

usage() {
	echo "Usage: run.sh <options> [ -- <additional arguments to test> ]"
	echo
	echo "  -h|--help                    Show this help message"
	echo "  -d|--device  <netdev>        Select which network device to use ($netdev)"
	echo "  -p|--port  <number>          Set the TCP port to use ($tcpport)"
	echo "  -m|--mode  <mode>            Which mode to use - tcp or ucx ($mode)"
	echo "  -w|--wait-time  <seconds>    How much time to wait before connecting ($wait_time)"
	echo "  -H|--hostlist  <hostlist>    Provide a hostlist for pdsh command"
	echo
}

while [[ $# -gt 0 ]]
do
	key="$1"
	echo $key
	case $key in
	-h|--help)
		usage
		exit 0
		;;
	-d|--device)
		netdev=$2
		shift
		;;
	--p|--port)
		tcpport=$2
		shift
		;;
	-m|--mode)
		mode=$2
		shift
		;;
	-w|--wait-time)
		wait_time=$2
		shift
		;;
	-H|--hostlist)
		pdsh_args="-w $2"
		shift
		;;
	--)
		shift
		break
		;;
	*)
		usage
		exit -2
		;;
	esac
	shift
done

addrlist=$(pdsh $pdsh_args "ip -o -4 a s $netdev" | sed -ne 's:.*inet \([0-9.]*\)/.*:\1:p')
for addr in $addrlist
do
	echo $addr $tcpport
done > $hostfile

set -x
pdsh $pdsh_args "$mydir/sa -p $tcpport -f $hostfile -w $wait_time -m $mode $@"
rm $hostfile
