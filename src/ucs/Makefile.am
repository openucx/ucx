#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
# Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

SUBDIRS = vfs/sock . vfs/fuse signal

AUTOMAKE_OPTIONS    = nostdinc # avoid collision with built-in debug.h
lib_LTLIBRARIES     = libucs.la
bin_PROGRAMS        =

libucs_la_CPPFLAGS = $(BASE_CPPFLAGS) $(BFD_CPPFLAGS) \
                     -DUCX_MODULE_DIR=\"$(moduledir)\" \
                     -DUCX_CONFIG_DIR=\"$(ucx_config_dir)\"
libucs_la_CFLAGS   = $(BASE_CFLAGS) $(BFD_CFLAGS) $(LT_CFLAGS)
libucs_la_LDFLAGS  = -ldl $(BFD_LDFLAGS) -version-info $(SOVERSION)
libucs_ladir       = $(includedir)/ucs
libucs_la_LIBADD   = $(LIBM) $(top_builddir)/src/ucm/libucm.la $(BFD_LIBS)

nobase_dist_libucs_la_HEADERS = \
	arch/aarch64/bitops.h \
	arch/ppc64/bitops.h \
	arch/rv64/bitops.h \
	arch/x86_64/bitops.h \
	arch/bitops.h \
	algorithm/crc.h \
	algorithm/qsort_r.h \
	algorithm/string_distance.h \
	async/async_fwd.h \
	config/global_opts.h \
	config/ini.h \
	config/parser.h \
	config/types.h \
	datastruct/array.h \
	datastruct/callbackq.h \
	datastruct/callbackq_compat.h \
	datastruct/hlist.h \
	datastruct/khash.h \
	datastruct/linear_func.h \
	datastruct/list.h \
	datastruct/mpool.h \
	datastruct/mpool_set.h \
	datastruct/pgtable.h \
	datastruct/piecewise_func.h \
	datastruct/queue_types.h \
	datastruct/strided_alloc.h \
	datastruct/string_buffer.h \
	datastruct/string_set.h \
	debug/log_def.h \
	debug/debug.h \
	debug/memtrack.h \
	memory/rcache.h \
	memory/memory_type.h \
	memory/memtype_cache.h \
	memory/numa.h \
	profile/profile_defs.h \
	profile/profile_off.h \
	profile/profile_on.h \
	stats/stats_fwd.h \
	stats/libstats.h \
	sys/event_set.h \
	sys/compiler_def.h\
	sys/math.h \
	sys/preprocessor.h \
	sys/string.h \
	sys/sock.h \
	sys/topo/base/topo.h \
	sys/stubs.h \
	sys/uid.h \
	time/time_def.h \
	type/class.h \
	type/param.h \
	type/init_once.h \
	type/spinlock.h \
	type/status.h \
	type/thread_mode.h \
	type/cpu_set.h \
	vfs/base/vfs_obj.h \
	vfs/base/vfs_cb.h \
	arch/atomic.h \
	arch/x86_64/global_opts.h \
	arch/x86_64/atomic.h \
	arch/aarch64/global_opts.h \
	arch/generic/atomic.h \
	arch/ppc64/global_opts.h \
	arch/rv64/global_opts.h \
	arch/global_opts.h

noinst_HEADERS = \
	arch/aarch64/cpu.h \
	arch/generic/cpu.h \
	arch/ppc64/cpu.h \
	arch/rv64/cpu.h \
	arch/x86_64/cpu.h \
	arch/cpu.h \
	config/ucm_opts.h \
	datastruct/arbiter.h \
	datastruct/bitmap.h \
	datastruct/dynamic_bitmap.h \
	datastruct/frag_list.h \
        datastruct/lru.h \
	datastruct/mpmc.h \
	datastruct/mpool.inl \
	datastruct/mpool_set.inl \
	datastruct/ptr_array.h \
	datastruct/queue.h \
	datastruct/sglib.h \
	datastruct/sglib_wrapper.h \
	datastruct/conn_match.h \
	datastruct/ptr_map.h \
	datastruct/ptr_map.inl \
	datastruct/static_bitmap.h \
	datastruct/usage_tracker.h \
	debug/assert.h \
	debug/debug_int.h \
	debug/log.h \
	debug/memtrack_int.h \
	memory/numa.h \
	memory/rcache_int.h \
	memory/rcache.inl \
	profile/profile.h \
	stats/stats.h \
	sys/checker.h \
	sys/compiler.h \
	sys/lib.h \
	sys/module.h \
	sys/sys.h \
	sys/iovec.h \
	sys/iovec.inl \
	sys/ptr_arith.h \
	time/time.h \
	time/timerq.h \
	time/timer_wheel.h \
	type/serialize.h \
	type/float8.h \
	async/async.h \
	async/pipe.h \
	async/eventfd.h \
	async/signal.h \
	async/thread.h \
	async/async_int.h

libucs_la_SOURCES = \
	algorithm/crc.c \
	algorithm/qsort_r.c \
	algorithm/string_distance.c \
	arch/aarch64/cpu.c \
	arch/aarch64/global_opts.c \
	arch/ppc64/timebase.c \
	arch/ppc64/global_opts.c \
	arch/rv64/cpu.c \
	arch/rv64/global_opts.c \
	arch/x86_64/cpu.c \
	arch/x86_64/global_opts.c \
	arch/cpu.c \
	async/async.c \
	async/signal.c \
	async/pipe.c \
	async/eventfd.c \
	async/thread.c \
	config/global_opts.c \
	config/ucm_opts.c \
	config/ini.c \
	config/parser.c \
	datastruct/arbiter.c \
	datastruct/array.c \
	datastruct/bitmap.c \
	datastruct/callbackq.c \
	datastruct/frag_list.c \
	datastruct/lru.c \
	datastruct/mpmc.c \
	datastruct/mpool.c \
	datastruct/mpool_set.c \
	datastruct/pgtable.c \
	datastruct/piecewise_func.c \
	datastruct/ptr_array.c \
	datastruct/ptr_map.c \
	datastruct/strided_alloc.c \
	datastruct/string_buffer.c \
	datastruct/string_set.c \
	datastruct/usage_tracker.c \
	datastruct/conn_match.c \
	debug/assert.c \
	debug/debug.c \
	debug/log.c \
	debug/memtrack.c \
	memory/memory_type.c \
	memory/memtype_cache.c \
	memory/numa.c \
	memory/rcache.c \
	memory/rcache_vfs.c \
	profile/profile.c \
	stats/stats.c \
	sys/event_set.c \
	sys/init.c \
	sys/math.c \
	sys/module.c \
	sys/string.c \
	sys/sys.c \
	sys/iovec.c \
	sys/lib.c \
	sys/sock.c \
	sys/topo/base/topo.c \
	sys/stubs.c \
	sys/uid.c \
	time/time.c \
	time/timer_wheel.c \
	time/timerq.c \
	type/class.c \
	type/status.c \
	type/spinlock.c \
	type/thread_mode.c \
	vfs/base/vfs_obj.c \
	vfs/base/vfs_cb.c

if HAVE_AARCH64_THUNDERX2
libucs_la_SOURCES += \
        arch/aarch64/memcpy_thunderx2.S
endif

if HAVE_STATS
libucs_la_SOURCES += \
	stats/client_server.c \
	stats/serialization.c \
	stats/libstats.c

bin_PROGRAMS            += ucs_stats_parser
ucs_stats_parser_CPPFLAGS = $(BASE_CPPFLAGS)
ucs_stats_parser_LDADD   = libucs.la
ucs_stats_parser_SOURCES = stats/stats_parser.c
endif

all-local: $(objdir)/$(modulesubdir)

$(objdir)/$(modulesubdir): $(lib_LTLIBRARIES)
	$(AM_V_at)$(LN_RS) -fn $(localmoduledir) $(objdir)/$(modulesubdir)

PKG_CONFIG_NAME=ucs

include $(top_srcdir)/config/module-pkg-config.am

#TODO	stats/stats_dump.c
#TODO	stats/stats_reader.c
