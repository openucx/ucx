#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_EFA

module_LTLIBRARIES        = libuct_ib_efa.la
libuct_ib_efa_la_CPPFLAGS = $(BASE_CPPFLAGS) $(IBVERBS_CPPFLAGS)
libuct_ib_efa_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libuct_ib_efa_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                            $(top_builddir)/src/uct/libuct.la \
                            $(top_builddir)/src/uct/ib/libuct_ib.la
libuct_ib_efa_la_LDFLAGS  = $(EFA_LIB) $(IBVERBS_LDFLAGS) \
                            -version-info $(SOVERSION)

libuct_ib_efa_la_SOURCES  = base/ib_efa_md.c \
                            srd/srd_iface.c

noinst_HEADERS            = base/ib_efa.h \
                            srd/srd_iface.h \
                            srd/srd_def.h

PKG_CONFIG_NAME=ib-efa

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am

endif # HAVE_EFA
