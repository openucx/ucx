#
# Copyright (C) Intel Corporation, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_GAUDI

SUBDIRS = .

module_LTLIBRARIES       = libuct_gaudi.la
libuct_gaudi_la_CPPFLAGS = $(BASE_CPPFLAGS) $(GAUDI_CPPFLAGS)
libuct_gaudi_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libuct_gaudi_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                           $(top_builddir)/src/uct/libuct.la
libuct_gaudi_la_LDFLAGS  = $(GAUDI_LDFLAGS) $(GAUDI_LIBS) -version-info $(SOVERSION)

noinst_HEADERS = \
    base/scal.h \
    base/gaudi_base.h \
    base/gaudi_topo.h

EXTRA_DIST =

libuct_gaudi_la_SOURCES = \
    base/gaudi_base.c \
    base/gaudi_topo.c

noinst_HEADERS += \
    gaudi_gdr/gaudi_gdr_md.h \
    gaudi_gdr/gaudi_gdr_iface.h

libuct_gaudi_la_SOURCES += \
    gaudi_gdr/gaudi_gdr_md.c \
    gaudi_gdr/gaudi_gdr_iface.c

PKG_CONFIG_NAME=gaudi

include $(top_srcdir)/config/module.am
# TODO: enable pkg-config processing when module static build is enabled
# include $(top_srcdir)/config/module-pkg-config.am

endif
