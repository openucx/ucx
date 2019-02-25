#
# Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_RDMACM

# rdmacm is under IB, but it's actually a uct module, because it defines its own
# memory domain component
module_LTLIBRARIES        = libuct_rdmacm.la
libuct_rdmacm_la_CPPFLAGS = $(BASE_CPPFLAGS) $(IBVERBS_CPPFLAGS) $(RDMACM_CPPFLAGS)
libuct_rdmacm_la_CFLAGS   = $(BASE_CFLAGS)
libuct_rdmacm_la_LIBADD   = $(RDMACM_LIBS)
libuct_rdmacm_la_LDFLAGS  = $(IBVERBS_LDFLAGS) $(RDMACM_LDFLAGS) -version-info $(SOVERSION)

noinst_HEADERS = \
	rdmacm_md.h \
	rdmacm_iface.h \
	rdmacm_ep.h \
	rdmacm_def.h \
	rdmacm_cm.h

libuct_rdmacm_la_SOURCES = \
	rdmacm_md.c \
	rdmacm_iface.c \
	rdmacm_ep.c \
	rdmacm_cm.c

include $(top_srcdir)/config/module.am

endif
