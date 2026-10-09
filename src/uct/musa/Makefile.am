#
# Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_MUSA

module_LTLIBRARIES      = libuct_musa.la
libuct_musa_la_CPPFLAGS = $(BASE_CPPFLAGS) $(MUSA_CPPFLAGS)
libuct_musa_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libuct_musa_la_LDFLAGS  = $(MUSA_LDFLAGS) -version-info $(SOVERSION) \
                          $(UCX_LT_RELEASE)
libuct_musa_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                          $(top_builddir)/src/uct/libuct.la \
                          $(MUSA_LIBS)
libuct_musa_ladir       = $(includedir)/uct/musa

noinst_HEADERS = \
	musa_utils.h \
	musa_base.h \
	musa_md.h \
	musa_copy_iface.h \
	musa_copy_ep.h

libuct_musa_la_SOURCES = \
	musa_base.c \
	musa_md.c \
	musa_copy_iface.c \
	musa_copy_ep.c

PKG_CONFIG_NAME=musa

include $(top_srcdir)/config/module.am

endif
