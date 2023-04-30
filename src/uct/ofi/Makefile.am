#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2022. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_OFI

module_LTLIBRARIES      = libuct_ofi.la
libuct_ofi_la_CPPFLAGS = $(BASE_CPPFLAGS)
libuct_ofi_la_CFLAGS   = $(BASE_CFLAGS) -I/opt/cray/libfabric/1.11.0.4.75/include/
libuct_ofi_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                          $(top_builddir)/src/uct/libuct.la
libuct_ofi_la_LDFLAGS  = -L/opt/cray/libfabric/1.11.0.4.75/lib64/ -lfabric -version-info $(SOVERSION)

noinst_HEADERS = \
	base/ofi_def.h \
	base/ofi_types.h \
	base/ofi_md.h \
	base/ofi_device.h \
	base/ofi_iface.h \
	base/ofi_ep.h

libuct_ofi_la_SOURCES = \
	base/ofi_md.c \
	base/ofi_device.c \
	base/ofi_iface.c \
	base/ofi_ep.c

PKG_CONFIG_NAME=ofi

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am

endif
