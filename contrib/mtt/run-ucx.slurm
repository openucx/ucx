#!/bin/sh
#SBATCH --job-name=ucx_oshmem
#SBATCH --ntasks-per-node=8

mydir=$SLURM_SUBMIT_DIR
cd $mydir

trap '' SIGTERM
trap '' SIGHUP
trap '' SIGINT

umask 022
ulimit -c 0

PROG=`basename $0`
HOST=`hostname`
TS=`date +%Y%m%d_%H%M%S`_$$_$SLURM_JOB_ID
MTT_HOME=${MTT_HOME:-~mtt/svn/mtt/trunk}
MTT_EXE=$MTT_HOME/client/mtt
MTT_OPT="-d -v"

if [ ! -x ${MTT_EXE} ];then
   echo "Unable to find mtt executable under ${MTT_HOME}"
exit 1
fi

mtt_ini_opts=""
mtt_ini_opts="$mtt_ini_opts $*"

MTT_SCRATCH=${MTT_SCRATCH:-${MTT_SCRATCH_ROOT}/ucx_oshmem/${TS}_${HOST}}

mkdir -p $MTT_SCRATCH

set -x
$MTT_EXE $MTT_OPT -s $MTT_SCRATCH -f $mydir/mtt_ucx_oshmem.ini $mtt_ini_opts 

sleep 10
