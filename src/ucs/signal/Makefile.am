#
# Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

lib_LTLIBRARIES = libucs_signal.la

libucs_signal_la_CPPFLAGS = $(BASE_CPPFLAGS)
libucs_signal_la_CFLAGS   = $(BASE_CFLAGS)
libucs_signal_la_LDFLAGS  = -version-info $(SOVERSION)
libucs_signal_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la
libucs_signal_la_SOURCES  = signal.c
