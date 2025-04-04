#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
#
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

CC = $(MPICC)
LD = $(MPICC)


# Test application for memory hooks when running with MPI
# (some MPIs have hooks of their own and we make sure ours still work)
noinst_PROGRAMS       =

if HAVE_MPICC
noinst_PROGRAMS      += test_memhooks
if HAVE_CUDA
noinst_PROGRAMS      += test_mpi_cuda
endif
endif

if HAVE_SHMEMCC
noinst_PROGRAMS      += shmem_pingpong
endif

AM_CPPFLAGS           = \
    $(BASE_CPPFLAGS) \
    -DUCS_LIB_DIR="$(abs_top_builddir)/src/ucs/.libs" \
    -DUCM_LIB_DIR="$(abs_top_builddir)/src/ucm/.libs" \
    -DTEST_LIB_DIR="$(abs_builddir)/.libs"
AM_CFLAGS             = $(BASE_CFLAGS)

if HAVE_MPICC
test_memhooks_SOURCES = test_memhooks.c
test_memhooks_LDFLAGS = -ldl

if HAVE_CUDA
test_mpi_cuda_SOURCES  = test_mpi_cuda.c
test_mpi_cuda_CPPFLAGS = $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
test_mpi_cuda_CFLAGS   = $(BASE_CFLAGS)
test_mpi_cuda_LDFLAGS  = $(CUDA_LDFLAGS)
test_mpi_cuda_LDADD    = $(CUDA_LIBS) $(top_builddir)/src/ucs/libucs.la
endif

# A library we use for testing that memory hooks work in libraries loaded
# after the hooks were installed
noinst_LTLIBRARIES = libtest_memhooks.la
libtest_memhooks_la_SOURCES = test_memhooks_lib.c
libtest_memhooks_la_LDFLAGS = -rpath /nowhere  # Force shared library
endif


if HAVE_SHMEMCC
# SHMEM ping-pong test
shmem_pingpong_LDFLAGS = -loshmem
shmem_pingpong_SOURCES = shmem_pingpong.c
endif
