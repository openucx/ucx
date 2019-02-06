#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_KNEM

module_LTLIBRARIES      = libuct_knem.la
libuct_knem_la_CFLAGS   = $(BASE_CFLAGS)
libuct_knem_la_CPPFLAGS = $(BASE_CPPFLAGS)
libuct_knem_la_LDFLAGS  = -version-info $(SOVERSION)

noinst_HEADERS = \
	knem_ep.h \
	knem_iface.h \
	knem_md.h

libuct_knem_la_SOURCES = \
    knem_ep.c \
    knem_iface.c \
    knem_md.c

include $(top_srcdir)/config/module.am

endif
