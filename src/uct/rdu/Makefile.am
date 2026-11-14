#
# Copyright (C) SambaNova Systems, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_RDU

module_LTLIBRARIES     = libuct_rdu.la
libuct_rdu_la_CPPFLAGS = $(BASE_CPPFLAGS)
libuct_rdu_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libuct_rdu_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                         $(top_builddir)/src/uct/libuct.la \
                         $(RDU_LIBS)
libuct_rdu_la_LDFLAGS  = -version-info $(SOVERSION) $(UCX_LT_RELEASE)

noinst_HEADERS = \
    base/rdu_md.h \
    base/uct_rdu_devmem.h

libuct_rdu_la_SOURCES = \
    base/rdu_md.c \
    base/uct_rdu_devmem.c

PKG_CONFIG_NAME=rdu

include $(top_srcdir)/config/module.am

endif
