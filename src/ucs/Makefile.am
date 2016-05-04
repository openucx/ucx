#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) 2014 Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AUTOMAKE_OPTIONS    = nostdinc # avoid collision with built-in debug.h
AM_CPPFLAGS         = -I$(top_builddir) -I$(abs_top_srcdir)/src
lib_LTLIBRARIES     = libucs.la
bin_PROGRAMS        =

libucs_la_LDFLAGS  = -ldl -version-info $(SOVERSION)
libucs_ladir       = $(includedir)/ucs
libucs_la_LIBADD   = \
	$(LIBM) \
	$(top_builddir)/src/ucm/libucm.la

nobase_dist_libucs_la_HEADERS = \
	algorithm/qsort_r.h \
	arch/aarch64/bitops.h \
	arch/aarch64/cpu.h \
	arch/generic/atomic.h \
	arch/ppc64/bitops.h \
	arch/ppc64/cpu.h \
	arch/x86_64/atomic.h \
	arch/x86_64/bitops.h \
	arch/x86_64/cpu.h \
	arch/atomic.h \
	arch/bitops.h \
	arch/cpu.h \
	async/async_fwd.h \
	async/async.h \
	async/pipe.h \
	async/signal.h \
	async/thread.h \
	config/global_opts.h \
	config/parser.h \
	config/types.h \
	datastruct/arbiter.h \
	datastruct/callbackq.h \
	datastruct/frag_list.h \
	datastruct/hash.h \
	datastruct/list.h \
	datastruct/mpmc.h \
	datastruct/mpool.h \
	datastruct/mpool.inl \
	datastruct/pgtable.h \
	datastruct/ptr_array.h \
	datastruct/queue_types.h \
	datastruct/queue.h \
	datastruct/sglib.h \
	datastruct/sglib_wrapper.h \
	debug/check.h \
	debug/debug.h \
	debug/instrument.h \
	debug/log.h \
	debug/memtrack.h \
	stats/libstats.h \
	stats/stats.h \
	sys/compiler.h \
	sys/math.h \
	sys/preprocessor.h \
	sys/rcache.h \
	sys/sys.h \
	time/time.h \
	time/timerq.h \
	time/timer_wheel.h \
	type/class.h \
	type/component.h \
	type/spinlock.h \
	type/status.h \
	type/thread_mode.h

noinst_HEADERS = \
	async/async_int.h

libucs_la_SOURCES = \
	algorithm/qsort_r.c \
	arch/ppc64/timebase.c \
	arch/x86_64/cpu.c \
	async/async.c \
	async/signal.c \
	async/pipe.c \
	async/thread.c \
	config/global_opts.c \
	config/parser.c \
	datastruct/arbiter.c \
	datastruct/callbackq.c \
	datastruct/frag_list.c \
	datastruct/mpmc.c \
	datastruct/mpool.c \
	datastruct/pgtable.c \
	datastruct/ptr_array.c \
	debug/debug.c \
	debug/instrument.c \
	debug/log.c \
	debug/memtrack.c \
	stats/stats.c \
	sys/init.c \
	sys/math.c \
	sys/rcache.c \
	sys/sys.c \
	time/time.c \
	time/timer_wheel.c \
	time/timerq.c \
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


#TODO	stats/stats_dump.c
#TODO	stats/stats_reader.c
