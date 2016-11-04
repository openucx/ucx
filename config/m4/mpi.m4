#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
#
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#


#
# Enable compiling tests with MPI
#
AC_ARG_WITH([mpi],
            [AS_HELP_STRING([--with-mpi@<:@=MPIHOME@:>@], [Compile MPI tests (default is NO).])],[:],[with_mpi=no])

    AS_IF([test "x$with_mpi" != xyes && test "x$with_mpi" != xno],
            [
            AS_IF([test -d "$with_mpi/bin"],[with_mpi="$with_mpi/bin"],[:])
            mpi_path=$with_mpi;with_mpi=yes
            ], 
            mpi_path=$PATH)

#
# Search for mpicc and mpirun in the given path.
#
AS_IF([test "x$with_mpi" == xyes],
        [
        AC_ARG_VAR(MPICC,[MPI C compiler command])
        AC_PATH_PROGS(MPICC,mpicc mpiicc,"",$mpi_path)
        AC_ARG_VAR(MPIRUN,[MPI launch command])
        AC_PATH_PROGS(MPIRUN,mpirun mpiexec aprun orterun,"",$mpi_path)
        AS_IF([test -z "$MPIRUN"], AC_MSG_ERROR([--with-mpi was requested but MPI was not found in the PATH in $mpi_path]),[:])
        ],[:])

AS_IF([test -n "$MPICC"], AC_DEFINE([HAVE_MPI], [1], [MPI support]))
AM_CONDITIONAL([HAVE_MPI],    [test -n "$MPIRUN"])
AM_CONDITIONAL([HAVE_MPICC],  [test -n "$MPICC"])
AM_CONDITIONAL([HAVE_MPIRUN], [test -n "$MPIRUN"])
