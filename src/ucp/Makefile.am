#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2017. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

lib_LTLIBRARIES     = libucp.la

libucp_la_CFLAGS   = $(BASE_CFLAGS)
libucp_la_LIBS     =
libucp_la_CPPFLAGS = $(BASE_CPPFLAGS) -I$(abs_top_srcdir)/src -I$(abs_top_builddir)/src
libucp_la_LDFLAGS  = -ldl -version-info $(SOVERSION)
libucp_la_LIBADD   = ../ucs/libucs.la ../uct/libuct.la
libucp_ladir       = $(includedir)/ucp

nobase_dist_libucp_la_HEADERS = \
	api/ucp_compat.h \
	api/ucp_def.h \
	api/ucp_version.h \
	api/ucp.h

noinst_HEADERS = \
	core/ucp_context.h \
	core/ucp_ep.h \
	core/ucp_ep.inl \
	core/ucp_listener.h \
	core/ucp_mm.h \
	core/ucp_proxy_ep.h \
	core/ucp_request.h \
	core/ucp_request.inl \
	core/ucp_worker.h \
	core/ucp_thread.h \
	core/ucp_types.h \
	dt/dt.h \
	dt/dt.inl \
	dt/dt_contig.h \
	dt/dt_iov.h \
	dt/dt_generic.h \
	proto/proto.h \
	proto/proto_am.inl \
	rma/rma.h \
	rma/rma.inl \
	tag/eager.h \
	tag/rndv.h \
	tag/tag_match.h \
	tag/tag_match.inl \
	tag/offload.h \
	wireup/address.h \
	wireup/ep_match.h \
	wireup/wireup_ep.h \
	wireup/wireup.h \
	stream/stream.h

libucp_la_SOURCES = \
	core/ucp_context.c \
	core/ucp_ep.c \
	core/ucp_listener.c \
	core/ucp_mm.c \
	core/ucp_proxy_ep.c \
	core/ucp_request.c \
	core/ucp_rkey.c \
	core/ucp_version.c \
	core/ucp_worker.c \
	dt/dt_contig.c \
	dt/dt_iov.c \
	dt/dt_generic.c \
	dt/dt.c \
	proto/proto_am.c \
	rma/amo_basic.c \
	rma/amo_send.c \
	rma/rma_basic.c \
	rma/rma_send.c \
	rma/rma_sw.c \
	rma/flush.c \
	tag/eager_rcv.c \
	tag/eager_snd.c \
	tag/probe.c \
	tag/rndv.c \
	tag/tag_match.c \
	tag/tag_recv.c \
	tag/tag_send.c \
	tag/offload.c \
	wireup/address.c \
	wireup/ep_match.c \
	wireup/select.c \
	wireup/signaling_ep.c \
	wireup/wireup_ep.c \
	wireup/wireup.c \
	stream/stream_send.c \
	stream/stream_recv.c

