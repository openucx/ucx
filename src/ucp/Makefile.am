#
# Copyright (C) Mellanox Technologies Ltd. 2001-2021.  ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2017. ALL RIGHTS RESERVED.
# Copyright (C) Los Alamos National Security, LLC. 2019. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

lib_LTLIBRARIES     = libucp.la

libucp_la_CFLAGS   = $(BASE_CFLAGS)
libucp_la_LIBS     =
libucp_la_CPPFLAGS = $(BASE_CPPFLAGS)
libucp_la_LDFLAGS  = -ldl -version-info $(SOVERSION)
libucp_la_LIBADD   = ../ucs/libucs.la ../uct/libuct.la
libucp_ladir       = $(includedir)/ucp

nobase_dist_libucp_la_HEADERS = \
	api/ucp_compat.h \
	api/ucp_def.h \
	api/ucp_version.h \
	api/ucp.h

noinst_HEADERS = \
	core/ucp_am.h \
	core/ucp_context.h \
	core/ucp_ep.h \
	core/ucp_ep.inl \
	core/ucp_listener.h \
	core/ucp_mm.h \
	core/ucp_proxy_ep.h \
	core/ucp_request.h \
	core/ucp_request.inl \
	core/ucp_rkey.h \
	core/ucp_rkey.inl \
	core/ucp_worker.h \
	core/ucp_worker.inl \
	core/ucp_thread.h \
	core/ucp_types.h \
	dt/datatype_iter.h \
	dt/datatype_iter.inl \
	dt/dt.h \
	dt/dt.inl \
	dt/dt_contig.h \
	dt/dt_iov.h \
	dt/dt_generic.h \
	proto/lane_type.h \
	proto/proto_am.h \
	proto/proto_am.inl \
	proto/proto_common.h \
	proto/proto_common.inl \
	proto/proto_multi.h \
	proto/proto_multi.inl \
	proto/proto_select.h \
	proto/proto_select.inl \
	proto/proto_single.h \
	proto/proto_single.inl \
	proto/proto.h \
	rma/rma.h \
	rma/rma.inl \
	rndv/proto_rndv.h \
	rndv/proto_rndv.inl \
	rndv/rndv.h \
	tag/eager.h \
	tag/tag_rndv.h \
	tag/tag_match.h \
	tag/tag_match.inl \
	tag/offload.h \
	wireup/address.h \
	wireup/ep_match.h \
	wireup/wireup_ep.h \
	wireup/wireup.h \
	wireup/wireup_cm.h \
	stream/stream.h

devel_headers = \
	core/ucp_resource.h

if INSTALL_DEVEL_HEADERS
nobase_dist_libucp_la_HEADERS += $(devel_headers)
else
noinst_HEADERS += $(devel_headers)
endif

if ENABLE_EXPERIMENTAL_API
nobase_dist_libucp_la_HEADERS += api/ucpx.h
else
noinst_HEADERS                += api/ucpx.h
endif

libucp_la_SOURCES = \
	core/ucp_context.c \
	core/ucp_am.c \
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
	proto/lane_type.c \
	proto/proto_am.c \
	proto/proto_common.c \
	proto/proto_reconfig.c \
	proto/proto_multi.c \
	proto/proto_select.c \
	proto/proto_single.c \
	proto/proto.c \
	rma/amo_basic.c \
	rma/amo_send.c \
	rma/amo_sw.c \
	rma/get_am.c \
	rma/get_offload.c \
	rma/put_am.c \
	rma/put_offload.c \
	rma/rma_basic.c \
	rma/rma_send.c \
	rma/rma_sw.c \
	rma/flush.c \
	rndv/proto_rndv.c \
	rndv/rndv_am.c \
	rndv/rndv_get.c \
	rndv/rndv_rtr.c \
	rndv/rndv.c \
	tag/eager_multi.c \
	tag/eager_rcv.c \
	tag/eager_snd.c \
	tag/eager_single.c \
	tag/probe.c \
	tag/tag_rndv.c \
	tag/tag_match.c \
	tag/tag_recv.c \
	tag/tag_send.c \
	tag/offload.c \
	tag/offload/eager.c \
	wireup/address.c \
	wireup/ep_match.c \
	wireup/select.c \
	wireup/wireup_ep.c \
	wireup/wireup.c \
	wireup/wireup_cm.c \
	stream/stream_send.c \
	stream/stream_recv.c

