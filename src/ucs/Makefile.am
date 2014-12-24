#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) 2014 Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
# $COPYRIGHT$
# $HEADER$
#

AUTOMAKE_OPTIONS    = nostdinc # avoid collision with built-in debug.h
AM_CPPFLAGS         = -I$(top_builddir) -I$(abs_top_srcdir)/src
lib_LTLIBRARIES     = libucs.la
bin_PROGRAMS        =

libucs_la_LDFLAGS  = -ldl -version-info $(SOVERSION)
libucs_la_LIBADD   = $(LIBM)
libucs_ladir       = $(includedir)

noinst_HEADERS = \
	config/global_opts.h \
	config/parser.h \
	config/types.h \
	datastruct/frag_list.h \
	datastruct/list.h \
	datastruct/mpmc.h \
	datastruct/mpool.h \
	datastruct/ptr_array.h \
	datastruct/queue.h \
	datastruct/sglib.h \
	datastruct/sglib_wrapper.h \
	debug/log.h \
	debug/check.h \
	debug/debug.h \
	debug/instrument.h \
	debug/memtrack.h \
	gtest/test_helpers.h \
	gtest/test.h \
	stats/stats.h \
	stats/libstats.h \
	sys/asm/asm-aarch64.h \
	sys/asm/asm-ppc64.h \
	sys/asm/asm-x86_64.h \
	sys/arch.h \
	sys/compiler.h \
	sys/math.h \
	sys/preprocessor.h \
	sys/sys.h \
	time/time.h \
	time/timer_wheel.h \
	time/timerq.h \
	type/callback.h \
	type/class.h \
	type/component.h \
	type/spinlock.h \
	type/status.h 
	
libucs_la_SOURCES = \
	config/global_opts.c \
	config/parser.c \
	datastruct/frag_list.c \
	datastruct/mpmc.c \
	datastruct/mpool.c \
	datastruct/ptr_array.c \
	debug/debug.c \
	debug/instrument.c \
	debug/log.c \
	debug/memtrack.c \
	stats/stats.c \
	sys/init.c \
	sys/math.c \
	sys/sys.c \
	time/time.c \
	time/timer_wheel.c \
	time/timerq.c \
	type/callback.c \
	type/class.c \
	type/component.c \
	type/status.c
	
if HAVE_STATS
libucs_la_SOURCES += \
	stats/client_server.c \
	stats/serialization.c \
	stats/libstats.c

bin_PROGRAMS            += ucs_stats_parser
ucs_stats_parser_LDADD   = libucs.la
ucs_stats_parser_SOURCES = stats/stats_parser.c
endif

if HAVE_GTEST
lib_LTLIBRARIES       += libucstest.la
libucstest_la_LDFLAGS  = -ldl -version-info \
			$(SOVERSION) \
			$(GTEST_LDFLAGS)
libucstest_la_CPPFLAGS = -I$(top_builddir) -I$(abs_top_srcdir)/src \
			 $(GTEST_CPPFLAGS)
libucstest_la_CXXFLAGS = $(GTEST_CXXFLAGS)
libucstest_la_LIBADD   = libucs.la
			$(GTEST_LIBS) \
			$(BOOST_THREAD_LIB)
libucstest_ladir       = $(includedir)

libucstest_la_SOURCES  = \
	gtest/main.cc \
	gtest/test_helpers.cc \
	gtest/test.cc
endif


#TODO	stats/stats_dump.c
#TODO	stats/stats_reader.c
