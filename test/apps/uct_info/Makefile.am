#
# Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if ENABLE_STATIC

noinst_PROGRAMS = uct_info uct_info_static

uct_info_CPPFLAGS = $(BASE_CPPFLAGS)

uct_info_static_CPPFLAGS = $(BASE_CPPFLAGS)

uct_info_CFLAGS = $(BASE_CFLAGS)

uct_info_static_CFLAGS = $(BASE_CFLAGS)

uct_info_SOURCES = \
	uct_info.c

uct_info_static_SOURCES = \
	uct_info.c

uct_info_LDADD = $(top_builddir)/src/ucs/libucs.la \
                 $(top_builddir)/src/uct/libuct.la

uct_info_static_LDFLAGS = \
	-L$(top_builddir)/src/uct/.libs \
	-L$(top_builddir)/src/ucs/.libs \
	-L$(top_builddir)/src/ucm/.libs

if HAVE_CMA
uct_info_static_LDFLAGS += \
	-Wl,--undefined=uct_cma_init -Wl,-Bstatic,-luct_cma,-Bdynamic \
	-L$(top_builddir)/src/uct/sm/scopy/cma/.libs
endif

if HAVE_KNEM
uct_info_static_LDFLAGS += \
	-Wl,--undefined=uct_knem_init -Wl,-Bstatic,-luct_knem,-Bdynamic \
	-L$(top_builddir)/src/uct/sm/scopy/knem/.libs
endif

if HAVE_XPMEM
uct_info_static_LDFLAGS += \
	-Wl,--undefined=uct_xpmem_init -Wl,-Bstatic,-luct_xpmem,-Bdynamic \
	-L$(top_builddir)/src/uct/sm/mm/xpmem/.libs $(XPMEM_LIBS)
endif # HAVE_XPMEM

uct_info_static_LDFLAGS += \
	-Wl,--undefined=uct_init -Wl,-Bstatic,-luct,-Bdynamic \
	-Wl,--undefined=ucs_init -Wl,-Bstatic,-lucs,-Bdynamic \
	-Wl,--undefined=ucm_init -Wl,-Bstatic,-lucm,-Bdynamic \
	$(BFD_LIBS) $(BFD_LDFLAGS) -ldl

endif
