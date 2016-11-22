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

libucm_la_CPPFLAGS = \
    -I$(abs_top_srcdir)/src \
    -I$(abs_top_builddir)/src \
    -DUCM_MALLOC_PREFIX=ucm_dl

libucm_la_CFLAGS = \
	$(CFLAGS_NO_DEPRECATED)

nobase_dist_libucm_la_HEADERS = \
	api/ucm.h

noinst_HEADERS = \
	event/event.h \
	malloc/malloc_hook.h \
	malloc/allocator.h \
	mmap/mmap.h \
	util/ucm_config.h \
	util/log.h \
	util/reloc.h \
	util/sys.h

libucm_la_SOURCES = \
	event/event.c \
	malloc/malloc_hook.c \
	mmap/install.c \
	mmap/replace.c \
	util/ucm_config.c \
	util/log.c \
	util/reloc.c \
	util/sys.c

if HAVE_UCM_PTMALLOC283
libucm_la_CPPFLAGS += \
    -I$(srcdir)/ptmalloc283/sysdeps/pthread \
    -I$(srcdir)/ptmalloc283/sysdeps/generic \
    -fno-strict-aliasing \
    -DUSE_TSD_DATA_HACK \
    -DUSE_LOCKS=1 \
    -DONLY_MSPACES=0 \
    -DMALLINFO_FIELD_TYPE=int

libucm_la_SOURCES += \
    ptmalloc283/malloc.c

noinst_HEADERS += \
    ptmalloc283/malloc-2.8.3.h \
    ptmalloc283/sysdeps/generic/atomic.h \
    ptmalloc283/sysdeps/generic/malloc-machine.h \
    ptmalloc283/sysdeps/generic/thread-st.h \
    ptmalloc283/sysdeps/pthread/malloc-machine.h \
    ptmalloc283/sysdeps/pthread/thread-st.h
endif

if HAVE_UCM_PTMALLOC286
libucm_la_CPPFLAGS += \
    -fno-strict-aliasing \
    -DUSE_LOCKS=1 \
    -DMALLINFO_FIELD_TYPE=int

libucm_la_SOURCES += \
    ptmalloc286/malloc.c

noinst_HEADERS += \
    ptmalloc286/malloc-2.8.6.h
endif
