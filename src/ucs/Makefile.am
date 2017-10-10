#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AUTOMAKE_OPTIONS    = nostdinc # avoid collision with built-in debug.h
AM_CPPFLAGS         = $(BASE_CPPFLAGS) -I$(top_builddir) -I$(abs_top_srcdir)/src
libucs_la_CFLAGS    = $(BASE_CFLAGS)
lib_LTLIBRARIES     = libucs.la
bin_PROGRAMS        =

libucs_la_LDFLAGS  = -ldl -version-info $(SOVERSION)
libucs_ladir       = $(includedir)/ucs
libucs_la_LIBADD   = \
	$(LIBM) \
	$(top_builddir)/src/ucm/libucm.la

nobase_dist_libucs_la_HEADERS = \
	algorithm/crc.h \
	algorithm/qsort_r.h \
	arch/aarch64/bitops.h \
	arch/aarch64/cpu.h \
	arch/generic/atomic.h \
	arch/generic/cpu.h \
	arch/ppc64/bitops.h \
	arch/ppc64/cpu.h \
	arch/x86_64/atomic.h \
	arch/x86_64/bitops.h \
	arch/x86_64/cpu.h \
	arch/atomic.h \
	arch/bitops.h \
	arch/cpu.h \
	async/async_fwd.h \
	config/global_opts.h \
	config/parser.h \
	config/types.h \
	datastruct/callbackq.h \
	datastruct/list_types.h \
	datastruct/mpool.h \
	debug/check.h \
	debug/debug.h \
	debug/log.h \
	debug/memtrack.h \
	debug/profile.h \
	stats/stats_fwd.h \
	stats/libstats.h \
	stats/stats.h \
	sys/compiler_def.h\
	sys/string.h \
	time/time_def.h \
	type/class.h \
	type/component.h \
	type/spinlock.h \
	type/status.h \
	type/thread_mode.h \
	type/cpu_set.h

noinst_HEADERS = \
	datastruct/arbiter.h \
	datastruct/frag_list.h \
	datastruct/hash.h \
	datastruct/list.h \
	datastruct/mpmc.h \
	datastruct/mpool.inl \
	datastruct/pgtable.h \
	datastruct/ptr_array.h \
	datastruct/queue_types.h \
	datastruct/queue.h \
	datastruct/sglib.h \
	datastruct/sglib_wrapper.h \
	datastruct/khash.h \
	sys/checker.h \
	sys/compiler.h \
	sys/math.h \
	sys/preprocessor.h \
	sys/rcache.h \
	sys/sys.h \
	time/time.h \
	time/timerq.h \
	time/timer_wheel.h \
	async/async.h \
	async/pipe.h \
	async/signal.h \
	async/thread.h \
	async/async_int.h

libucs_la_SOURCES = \
	algorithm/crc.c \
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
	debug/log.c \
	debug/memtrack.c \
	debug/profile.c \
	stats/stats.c \
	sys/init.c \
	sys/math.c \
	sys/rcache.c \
	sys/string.c \
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
