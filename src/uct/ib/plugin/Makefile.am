#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_IB

# Public header installed for plugin implementors
pluginincludedir      = $(includedir)/uct/ib/plugin
plugininclude_HEADERS = uct_ib_plugin.h

noinst_HEADERS = uct_ib_plugin.h

# Convenience library: weak stubs linked into every transport that uses the
# plugin API.  The real libuct_ib_plugin.so (if installed) overrides these at
# runtime via the two-pass RTLD_GLOBAL load in ucs_module_load_from_dir().
noinst_LTLIBRARIES                 = libuct_ib_plugin_stubs.la
libuct_ib_plugin_stubs_la_SOURCES  = uct_ib_plugin_stubs.c
libuct_ib_plugin_stubs_la_CPPFLAGS = $(BASE_CPPFLAGS)
libuct_ib_plugin_stubs_la_CFLAGS   = $(BASE_CFLAGS)

endif # HAVE_IB
