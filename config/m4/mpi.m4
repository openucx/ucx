#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
#
# $COPYRIGHT$
# $HEADER$
#


#
# Enable compiling tests with MPI
#
AS_IF([test -n "$MPI_HOME"], [with_ompi="$MPI_HOME"])
AC_ARG_WITH([mpi],
            [AS_HELP_STRING([--with-mpi@<:@=MPIHOME@:>@], [Compile MPI tests (default is NO).])],
            [],
            [with_mpi=no])

#
# Search for mpicc and mpirun in the given path.
#
mpirun_path=""
mpicc_path=""
AS_IF([test "x$with_mpi" != xno],
      [AS_IF([test "x$with_mpi" == xyes], 
              [AC_PATH_PROGS([mpicc_path],  [mpicc])
               AC_PATH_PROGS([mpirun_path], [mpirun])],
              [AC_PATH_PROGS([mpicc_path],  [mpicc],  [], [$with_mpi/bin])
               AC_PATH_PROGS([mpirun_path], [mpirun], [], [$with_mpi/bin])])
       AS_IF([! test -z $mpicc_path],
             [AC_DEFINE(HAVE_MPI, [1], "MPI compilation support")
              AC_SUBST([MPICC], [$mpicc_path])],
             [AC_MSG_ERROR(MPI support requsted but mpicc was not found)])
       AS_IF([! test -z $mpirun_path],
             [AC_SUBST([MPIRUN], [$mpirun_path])],
             [AC_MSG_ERROR(MPI support requsted but mpirun was not found)])
       ])

AM_CONDITIONAL([HAVE_MPI],    [! test -z $mpicc_path])
AM_CONDITIONAL([HAVE_MPIRUN], [! test -z $mpirun_path])