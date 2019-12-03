#
# Copyright (c) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

SUBDIRS = base builtin hicoll

lib_LTLIBRARIES    = libucg.la
libucg_la_CFLAGS   = $(BASE_CFLAGS)
libucg_la_CPPFLAGS = $(BASE_CPPFLAGS)
libucg_la_LDFLAGS  = -ldl -version-info $(SOVERSION)
libucg_ladir       = $(includedir)/ucg
libucg_la_SOURCES  =
libucg_la_LIBADD   = \
	../ucs/libucs.la \
	../uct/libuct.la \
	../ucp/libucp.la \
	base/libucg_base.la \
	builtin/libucg_builtin.la

nobase_dist_libucg_la_HEADERS = \
	api/ucg_def.h \
	api/ucg_version.h \
	api/ucg_plan_component.h \
	api/ucg_mpi.h \
	api/ucg.h
