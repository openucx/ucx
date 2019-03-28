#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
# Copyright (c) The University of Tennesse and the University of Tennessee
#               Research Foundation. 2016.  ALL RIGHTS RESERVED.
# Copyright (c) NVIDIA Corporation 2019. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

SUBDIRS = . cuda ib rocm sm ugni

lib_LTLIBRARIES    = libuct.la
libuct_la_CFLAGS   = $(BASE_CFLAGS)
libuct_la_CPPFLAGS = $(BASE_CPPFLAGS)
libuct_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la
libuct_la_LDFLAGS  = -ldl -version-info $(SOVERSION)
libuct_ladir       = $(includedir)/uct

nobase_dist_libuct_la_HEADERS = \
	api/tl.h \
	api/uct_def.h \
	api/uct.h \
	api/version.h

noinst_HEADERS = \
	base/uct_md.h \
	base/uct_iface.h \
	base/uct_log.h \
	base/uct_worker.h \
	sm/base/sm_ep.h \
	sm/base/sm_iface.h \
	sm/mm/base/mm_iface.h \
	sm/mm/base/mm_ep.h \
	sm/mm/base/mm_def.h \
	sm/mm/base/mm_md.h \
	sm/self/self.h \
	tcp/tcp.h \
	tcp/tcpcm/tcpcm_md.h \
	tcp/tcpcm/tcpcm_iface.h \
	tcp/tcpcm/tcpcm_ep.h \
	tcp/tcpcm/tcpcm_def.h

libuct_la_SOURCES = \
	base/uct_md.c \
	base/uct_mem.c \
	base/uct_iface.c \
	base/uct_worker.c \
	sm/base/sm_ep.c \
	sm/base/sm_iface.c \
	sm/mm/base/mm_iface.c \
	sm/mm/base/mm_ep.c \
	sm/mm/base/mm_md.c \
	sm/mm/posix/mm_posix.c \
	sm/mm/sysv/mm_sysv.c \
	sm/self/self.c \
	tcp/tcp_ep.c \
	tcp/tcp_iface.c \
	tcp/tcp_md.c \
	tcp/tcp_net.c \
	tcp/tcp_cm.c \
	tcp/tcpcm/tcpcm_md.c \
	tcp/tcpcm/tcpcm_iface.c \
	tcp/tcpcm/tcpcm_ep.c
