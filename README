<a href="https://scan.coverity.com/projects/5820">
  <img alt="Coverity Scan Build Status"
       src="https://scan.coverity.com/projects/5820/badge.svg"/>
</a>

UCX is a communication library implementing high-performance messaging for MPI/PGAS frameworks

### Quick start

% export PATH=$PATH:$MPI_HOME/bin  
% ./autogen.sh  
% ./contrib/configure-release --prefix=$PWD/install --with-mpi  
% make -j8 install  
% salloc -N2 --ntasks-per-node=1 mpirun --display-map $PWD/install/bin/ucx_perftest -d mlx5_1:1 -x rc_mlx5 -c 12 -t put_lat  

### UCX layout

UCX - Unified Communication X
UCP - UCX Protocol
UCT - UCX Transport
UCS - UCX Services

