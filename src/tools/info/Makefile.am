#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) The University of Tennessee and the University of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

bin_PROGRAMS          = ucx_info
AM_CPPFLAGS           = \
    $(BASE_CPPFLAGS) \
    -I$(abs_top_srcdir)/src \
    -I$(abs_top_builddir)/src
if HAVE_IB
AM_CPPFLAGS += $(IBVERBS_CPPFLAGS)
endif

BUILT_SOURCES = build_config.h
DISTCLEANFILES = build_config.h

#
# Produce a C header file which contains all defined variables from config.h
#
build_config.h: $(top_builddir)/config.h Makefile
	$(SED) -nr 's:\s*#define\s+(\w+)(\s+(\w+)|\s+(".*")|\s*)$$:{"\1", UCS_PP_MAKE_STRING(\3\4)},:p' <$(top_builddir)/config.h >$@

ucx_info_CFLAGS = $(BASE_CFLAGS)
if HAVE_CRAY_UGNI
ucx_info_CFLAGS += $(CRAY_UGNI_CFLAGS)
endif

ucx_info_SOURCES  = \
	build_info.c \
	proto_info.c \
	sys_info.c \
	tl_info.c \
	type_info.c \
	ucx_info.c

noinst_HEADERS = \
	build_config.h \
	ucx_info.h

ucx_info_LDADD    = \
    $(abs_top_builddir)/src/uct/libuct.la \
    $(abs_top_builddir)/src/ucp/libucp.la \
    $(abs_top_builddir)/src/ucs/libucs.la \
    $(abs_top_builddir)/src/ucm/libucm.la
