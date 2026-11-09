#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

module_LTLIBRARIES         = libtest_module.la
libtest_module_la_CPPFLAGS = $(BASE_CPPFLAGS)
libtest_module_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libtest_module_la_LDFLAGS  = -version-info $(SOVERSION) $(UCX_LT_RELEASE)
libtest_module_la_SOURCES  = test_module.c

module_LTLIBRARIES += libtestorder_base.la libtestorder_dependent.la \
                     libtestorder_extra.la

libtestorder_base_la_CPPFLAGS = $(BASE_CPPFLAGS)
libtestorder_base_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libtestorder_base_la_LDFLAGS  = -version-info $(SOVERSION) $(UCX_LT_RELEASE)
libtestorder_base_la_SOURCES  = test_module_base.c

libtestorder_dependent_la_CPPFLAGS = $(BASE_CPPFLAGS)
libtestorder_dependent_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libtestorder_dependent_la_LDFLAGS  = -version-info $(SOVERSION) \
                                   $(UCX_LT_RELEASE) \
                                   -Wl,--enable-new-dtags \
                                   -Wl,-rpath,$(abs_builddir)/.libs
libtestorder_dependent_la_LIBADD   = libtestorder_base.la
libtestorder_dependent_la_SOURCES  = test_module_dependent.c

libtestorder_extra_la_CPPFLAGS = $(BASE_CPPFLAGS)
libtestorder_extra_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libtestorder_extra_la_LDFLAGS  = -version-info $(SOVERSION) $(UCX_LT_RELEASE)
libtestorder_extra_la_SOURCES  = test_module_extra.c

include $(top_srcdir)/config/module.am

# disable install
install:
