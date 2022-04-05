#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# See file LICENSE for terms.
#

if HAVE_EFA_DV

module_LTLIBRARIES        = libuct_ib_efa.la
libuct_ib_efa_la_CFLAGS   = $(BASE_CFLAGS) $(IBVERBS_CFLAGS) $(EFA_CFLAGS)
libuct_ib_efa_la_CPPFLAGS = $(BASE_CPPFLAGS) $(IBVERBS_CPPFLAGS) $(EFA_CPPFLAGS)
libuct_ib_efa_la_LDFLAGS  = $(IBVERBS_LDFLAGS) $(EFA_LDFLAGS) -version-info $(SOVERSION)
libuct_ib_efa_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                            $(top_builddir)/src/uct/ib/libuct_ib.la \
							$(IBVERBS_LIBS) $(EFA_LIBS)

noinst_HEADERS = \
	ib_efa.h

libuct_ib_efa_la_SOURCES = \
	ib_efa_dv.c \
	ib_efa_md.c

if HAVE_TL_SRD
noinst_HEADERS += \
    ../srd/srd_def.h \
    ../srd/srd_ep.h \
    ../srd/srd_iface.h \
    ../srd/srd_inl.h

libuct_ib_efa_la_SOURCES += \
    ../srd/srd_ep.c \
    ../srd/srd_iface.c \
    ../srd/srd_log.c
endif # HAVE_TL_SRD

include $(top_srcdir)/config/module.am

endif # HAVE_EFA_DV
