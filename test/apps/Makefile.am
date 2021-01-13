#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2017.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

SUBDIRS = profiling

if HAVE_GLIBCXX_NOTHROW
SUBDIRS += iodemo
endif

if HAVE_CXX11
SUBDIRS += sockaddr
endif

noinst_PROGRAMS = \
	test_ucp_dlopen \
	test_ucs_dlopen \
	test_link_map \
	test_dlopen_cfg_print \
	test_init_mt

objdir = $(shell sed -n -e 's/^objdir=\(.*\)$$/\1/p' $(LIBTOOL))

test_ucs_dlopen_SOURCES  = test_ucs_dlopen.c
test_ucs_dlopen_CPPFLAGS = $(BASE_CPPFLAGS) \
                           -DLIB_PATH=$(abs_top_builddir)/src/ucs/$(objdir)/libucs.so
test_ucs_dlopen_CFLAGS   = $(BASE_CFLAGS)
test_ucs_dlopen_LDADD    = -ldl

test_ucp_dlopen_SOURCES  = test_ucp_dlopen.c
test_ucp_dlopen_CPPFLAGS = $(BASE_CPPFLAGS) \
                           -DLIB_PATH=$(abs_top_builddir)/src/ucp/$(objdir)/libucp.so
test_ucp_dlopen_CFLAGS   = $(BASE_CFLAGS)
test_ucp_dlopen_LDADD    = -ldl

test_link_map_SOURCES  = test_link_map.c
test_link_map_CPPFLAGS = $(BASE_CPPFLAGS)
test_link_map_CFLAGS   = $(BASE_CFLAGS)
test_link_map_LDADD    = -ldl $(top_builddir)/src/ucp/libucp.la

test_dlopen_cfg_print_SOURCES  = test_dlopen_cfg_print.c
test_dlopen_cfg_print_CPPFLAGS = $(BASE_CPPFLAGS) -g \
	-DUCS_LIB_PATH=$(abs_top_builddir)/src/ucs/$(objdir)/libucs.so \
	-DUCT_LIB_PATH=$(abs_top_builddir)/src/uct/$(objdir)/libuct.so
test_dlopen_cfg_print_CFLAGS   = $(BASE_CFLAGS)
test_dlopen_cfg_print_LDADD    = -ldl

test_init_mt_SOURCES  = test_init_mt.c
test_init_mt_CPPFLAGS = $(BASE_CPPFLAGS)
test_init_mt_CFLAGS   = $(BASE_CFLAGS) $(OPENMP_CFLAGS)
test_init_mt_LDADD    = $(top_builddir)/src/ucp/libucp.la

if HAVE_CUDA
noinst_PROGRAMS                += test_cuda_hook_dynamic

test_cuda_hook_dynamic_SOURCES  = test_cuda_hook.c
test_cuda_hook_dynamic_CPPFLAGS = $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
test_cuda_hook_dynamic_CFLAGS   = $(BASE_CFLAGS)
test_cuda_hook_dynamic_LDFLAGS  = $(CUDA_LDFLAGS)
test_cuda_hook_dynamic_LDADD    = $(top_builddir)/src/ucp/libucp.la \
                                  $(top_builddir)/src/ucm/libucm.la \
                                  $(CUDA_LIBS)

if HAVE_CUDA_STATIC
noinst_PROGRAMS                += test_cuda_hook_static
test_cuda_hook_static_SOURCES   = test_cuda_hook.c
test_cuda_hook_static_CPPFLAGS  = $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
test_cuda_hook_static_CFLAGS    = $(BASE_CFLAGS)
test_cuda_hook_static_LDFLAGS   = $(CUDA_LDFLAGS)
test_cuda_hook_static_LDADD     = $(top_builddir)/src/ucp/libucp.la \
                                  $(top_builddir)/src/ucm/libucm.la \
                                  $(CUDA_STATIC_LIBS) -lcuda -lrt -ldl -lpthread
endif

endif

if HAVE_TCMALLOC
noinst_PROGRAMS       += test_tcmalloc
test_tcmalloc_SOURCES  = test_tcmalloc.c
test_tcmalloc_CPPFLAGS = $(BASE_CPPFLAGS)
test_tcmalloc_CFLAGS   = $(BASE_CFLAGS)
test_tcmalloc_LDADD    = -ldl $(TCMALLOC_LIB) \
                          $(top_builddir)/src/ucp/libucp.la
endif
