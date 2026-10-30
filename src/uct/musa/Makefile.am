#
# Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
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
	base/musa_utils.h \
	base/musa_base.h

libuct_musa_la_SOURCES = \
	base/musa_base.c

noinst_HEADERS += \
	musa_copy/musa_copy_iface.h \
	musa_copy/musa_copy_ep.h

libuct_musa_la_SOURCES += \
	musa_copy/musa_copy_md.c \
	musa_copy/musa_copy_iface.c \
	musa_copy/musa_copy_ep.c

include $(top_srcdir)/config/module.am

endif
