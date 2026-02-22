#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_IB

# Build plugin first so libuct_ib.la can link against it
SUBDIRS = plugin . mlx5 efa rdmacm

module_LTLIBRARIES    = libuct_ib.la
libuct_ib_la_CPPFLAGS = $(BASE_CPPFLAGS) $(IBVERBS_CPPFLAGS)
libuct_ib_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libuct_ib_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                        $(top_builddir)/src/uct/libuct.la
libuct_ib_la_LDFLAGS  = $(IBVERBS_LDFLAGS) -version-info $(SOVERSION)

noinst_HEADERS = \
	base/ib_device.h \
	base/ib_iface.h \
	base/ib_log.h \
	base/ib_log.inl \
	base/ib_md.h \
	base/ib_verbs.h

libuct_ib_la_SOURCES = \
	base/ib_device.c \
	base/ib_iface.c \
	base/ib_log.c \
	base/ib_md.c

if HAVE_TL_RC
noinst_HEADERS += \
	rc/base/rc_def.h \
	rc/base/rc_ep.h \
	rc/base/rc_iface.h \
	rc/verbs/rc_verbs.h \
	rc/verbs/rc_verbs_impl.h

libuct_ib_la_SOURCES += \
	rc/base/rc_ep.c \
	rc/base/rc_iface.c \
	rc/verbs/rc_verbs_ep.c \
	rc/verbs/rc_verbs_iface.c

endif # HAVE_TL_RC

if HAVE_TL_UD
noinst_HEADERS += \
	ud/base/ud_iface_common.h \
	ud/base/ud_iface.h \
	ud/base/ud_ep.h \
	ud/base/ud_def.h \
	ud/base/ud_inl.h \
	ud/verbs/ud_verbs.h

libuct_ib_la_SOURCES += \
	ud/base/ud_iface_common.c \
	ud/base/ud_iface.c \
	ud/base/ud_ep.c \
	ud/base/ud_log.c \
	ud/verbs/ud_verbs.c

endif # HAVE_TL_UD

# Plugin stub implementation - always included in libuct_ib.so
# so that plugin functions are always available. The plugin library
# can override these when loaded.
libuct_ib_la_SOURCES += \
	plugin/base/uct_ib_plugin_stub.c

# Link against plugin library if available so it loads automatically
# and overrides weak symbols
# Plugin is built in subdirectory before this directory
# Use -Wl,--no-as-needed to force linking even though symbols are weak
if HAVE_IB_PLUGIN
libuct_ib_la_LDFLAGS += -Wl,--no-as-needed
libuct_ib_la_LIBADD += $(top_builddir)/src/uct/ib/plugin/libuct_ib_plugin.la
libuct_ib_la_LDFLAGS += -Wl,--as-needed
endif # HAVE_IB_PLUGIN

PKG_CONFIG_NAME=ib

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am

endif # HAVE_IB
