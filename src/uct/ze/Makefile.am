#
# Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_ZE

SUBDIRS = .

module_LTLIBRARIES    = libuct_ze.la
libuct_ze_la_CPPFLAGS = $(BASE_CPPFLAGS) $(ZE_CPPFLAGS)
libuct_ze_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libuct_ze_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                          $(top_builddir)/src/uct/libuct.la
libuct_ze_la_LDFLAGS  = $(ZE_LDFLAGS) $(ZE_LIBS) -version-info $(SOVERSION) \
                          $(patsubst %, -Xlinker %, -L$(ZE_ROOT)/lib -rpath $(ZE_ROOT)/hip/lib -rpath $(ZE_ROOT)/lib) \
                          $(patsubst %, -Xlinker %, --enable-new-dtags) \
                          $(patsubst %, -Xlinker %, -rpath $(ZE_ROOT)/lib64)

noinst_HEADERS = \
	base/ze_base.h

libuct_ze_la_SOURCES = \
	base/ze_base.c

noinst_HEADERS += \
    copy/ze_copy_md.h \
    copy/ze_copy_iface.h \
    copy/ze_copy_ep.h

libuct_ze_la_SOURCES += \
    copy/ze_copy_md.c \
    copy/ze_copy_iface.c \
    copy/ze_copy_ep.c

PKG_CONFIG_NAME=ze

include $(top_srcdir)/config/module.am
# TODO: enable pkg-config processing when module static build is enabled
# include $(top_srcdir)/config/module-pkg-config.am

endif
