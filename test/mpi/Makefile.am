#
# Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
#
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

CC = $(MPICC)
LD = $(MPICC)

# Test application for memory hooks when running with MPI
# (some MPIs have hooks of their own and we make sure ours still work)
noinst_PROGRAMS       = test_memhooks shmem_pingpong
AM_CPPFLAGS           = \
    $(BASE_CPPFLAGS) \
    -I$(abs_top_srcdir)/src \
    -I$(abs_top_builddir)/src \
    -DUCM_LIB_DIR="$(abs_top_builddir)/src/ucm/.libs" \
    -DTEST_LIB_DIR="$(abs_builddir)/.libs"
AM_CFLAGS             = $(BASE_CFLAGS)
test_memhooks_SOURCES = test_memhooks.c


# A library we use for testing that memory hooks work in libraries loaded
# after the hooks were installed
noinst_LTLIBRARIES = libtest_memhooks.la
libtest_memhooks_la_SOURCES = test_memhooks_lib.c
libtest_memhooks_la_LDFLAGS = -rpath /nowhere  # Force shared library


# SHMEM ping-pong test
shmem_pingpong_LDFLAGS = -loshmem
shmem_pingpong_SOURCES = shmem_pingpong.c
