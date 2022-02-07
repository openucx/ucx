#
# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_NVML

module_LTLIBRARIES     = libucs_nvml.la
libucs_nvml_la_CPPFLAGS = $(NVML_CPPFLAGS) $(BASE_CPPFLAGS)
libucs_nvml_la_CFLAGS   = $(NVML_CFLAGS) $(BASE_CFLAGS)
libucs_nvml_la_LIBADD   = $(NVML_LIBS) \
                         $(top_builddir)/src/ucs/libucs.la
libucs_nvml_la_LDFLAGS  = $(NVML_LDFLAGS) -version-info $(SOVERSION)
libucs_nvml_la_SOURCES  = topo_nvml.c
include $(top_srcdir)/config/module.am

endif
