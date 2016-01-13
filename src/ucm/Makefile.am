#
# Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

lib_LTLIBRARIES = libucm.la

libucm_ladir       = $(includedir)/ucm
libucm_la_LIBADD   = # ../ucs/libucs.la
libucm_la_LDFLAGS  = -Xlinker -z -Xlinker nodelete \
                     -Xlinker -z -Xlinker interpose

ptmalloc_cppflags = \
	-I$(srcdir)/ptmalloc3/sysdeps/pthread \
	-I$(srcdir)/ptmalloc3/sysdeps/generic \
	-fno-strict-aliasing \
	-DUSE_TSD_DATA_HACK \
	-DUSE_LOCKS=1 \
	-DONLY_MSPACES=0 \
	-DUSE_DL_PREFIX=1 \
	-DMALLINFO_FIELD_TYPE=int

libucm_la_CPPFLAGS = \
    -I$(abs_top_srcdir)/src \
    -I$(abs_top_builddir)/src \
    $(ptmalloc_cppflags)

libucm_la_CFLAGS = \
	-Wno-deprecated-declarations

nobase_dist_libucm_la_HEADERS = \
	api/ucm.h

ptmalloc3_sysdep_headers = \
	ptmalloc3/sysdeps/generic/atomic.h \
	ptmalloc3/sysdeps/generic/malloc-machine.h \
	ptmalloc3/sysdeps/generic/thread-st.h \
	ptmalloc3/sysdeps/pthread/malloc-machine.h \
	ptmalloc3/sysdeps/pthread/thread-st.h

noinst_HEADERS = \
	event/event.h \
	malloc/malloc_hook.h \
	mmap/mmap.h \
	util/ucm_config.h \
	util/log.h \
	util/reloc.h \
	ptmalloc3/malloc-2.8.3.h \
	$(ptmalloc3_sysdep_headers)

libucm_la_SOURCES = \
	event/event.c \
	malloc/malloc_hook.c \
	mmap/install.c \
	mmap/replace.c \
	ptmalloc3/malloc.c \
	util/ucm_config.c \
	util/log.c \
	util/reloc.c
	