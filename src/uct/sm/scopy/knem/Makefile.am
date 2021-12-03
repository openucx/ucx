#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_KNEM

module_LTLIBRARIES      = libuct_knem.la
libuct_knem_la_CFLAGS   = $(BASE_CFLAGS)
libuct_knem_la_CPPFLAGS = $(BASE_CPPFLAGS) $(KNEM_CPPFLAGS)
libuct_knem_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                          $(top_builddir)/src/uct/libuct.la
libuct_knem_la_LDFLAGS  = -version-info $(SOVERSION) $(UCT_MODULE_LDFLAGS)

noinst_HEADERS = \
	knem_ep.h \
	knem_iface.h \
	knem_md.h

libuct_knem_la_SOURCES = \
    knem_ep.c \
    knem_iface.c \
    knem_md.c

PKG_CONFIG_NAME=knem

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am

endif
