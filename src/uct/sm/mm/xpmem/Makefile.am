#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# Copyright (C) UChicago Argonne, LLC. 2019.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_XPMEM

module_LTLIBRARIES       = libuct_xpmem.la
libuct_xpmem_la_CFLAGS   = $(BASE_CFLAGS) $(XPMEM_CFLAGS)
libuct_xpmem_la_CPPFLAGS = $(BASE_CPPFLAGS)
libuct_xpmem_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                           $(top_builddir)/src/uct/libuct.la
libuct_xpmem_la_LDFLAGS  = $(XPMEM_LIBS) -version-info $(SOVERSION)
libuct_xpmem_la_SOURCES  = mm_xpmem.c

PKG_CONFIG_NAME=xpmem

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am

endif
