#
# Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if ENABLE_STATIC

noinst_PROGRAMS = uct_info uct_info_static

uct_info_CPPFLAGS = $(BASE_CPPFLAGS)

uct_info_static_CPPFLAGS = $(BASE_CPPFLAGS)

uct_info_SOURCES = \
	uct_info.c

uct_info_static_SOURCES = \
	uct_info.c

uct_info_LDADD = $(top_builddir)/src/ucs/libucs.la \
                 $(top_builddir)/src/uct/libuct.la

uct_info_static_LDFLAGS = \
	-Wl,--undefined=uct_cma_init -l:libuct_cma.a -L$(top_builddir)/src/uct/sm/scopy/cma/.libs \
	-Wl,--undefined=uct_knem_init -l:libuct_knem.a -L$(top_builddir)/src/uct/sm/scopy/knem/.libs

if HAVE_XPMEM
uct_info_static_LDFLAGS += \
	-Wl,--undefined=uct_xpmem_init -l:libuct_xpmem.a -L$(top_builddir)/src/uct/sm/mm/xpmem/.libs $(XPMEM_LIBS)
endif # HAVE_XPMEM

uct_info_static_LDFLAGS += \
	-Wl,--undefined=uct_init -L$(top_builddir)/src/uct/.libs -l:libuct.a \
	-Wl,--undefined=ucs_init -L$(top_builddir)/src/ucs/.libs -l:libucs.a \
	-Wl,--undefined=ucm_init -L$(top_builddir)/src/ucm/.libs -l:libucm.a \
    $(BFD_LIBS) $(BFD_LDFLAGS)

endif
