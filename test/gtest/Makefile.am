#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# Copyright (C) The University of Tennessee and the University of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
# Copyright (C) Los Alamos National Security, LLC. 2018 ALL RIGHTS RESERVED.
# Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2020.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_GTEST

# Set default configuration for running tests
UCX_HANDLE_ERRORS        ?= freeze
UCX_LOG_LEVEL            ?= warn
UCX_LOG_PRINT_ENABLE     ?= y
GTEST_FILTER             ?= *
GTEST_EXTRA_ARGS         ?=
LAUNCHER                 ?=
VALGRIND_EXTRA_ARGS      ?=

SUBDIRS = ucs/test_module ucm/test_dlopen

export UCX_HANDLE_ERRORS
export UCX_LOG_LEVEL
export UCX_LOG_PRINT_ENABLE

GTEST_ARGS = \
	--gtest_filter=$(GTEST_FILTER) \
	$(GTEST_EXTRA_ARGS)

VALGRIND_ARGS = \
	--tool=memcheck \
	--leak-check=full \
	--track-origins=yes \
	--fair-sched=try \
	--num-callers=25 \
	--error-exitcode=1 \
	--child-silent-after-fork=yes \
	--suppressions=$(top_srcdir)/contrib/valgrind.supp \
	$(VALGRIND_EXTRA_ARGS)

noinst_PROGRAMS = gtest

gtestdir  = $(includedir)
gtest_LDADD = \
	$(top_builddir)/src/ucs/libucs.la \
	$(top_builddir)/src/uct/libuct.la \
	$(top_builddir)/src/ucm/libucm.la \
	$(top_builddir)/src/ucp/libucp.la \
	$(top_builddir)/src/tools/perf/lib/libucxperf.la \
	$(OPENMP_CFLAGS) \
	$(GTEST_LIBS)


gtest_CPPFLAGS = \
	$(BASE_CPPFLAGS) \
	-I$(top_srcdir)/src \
	-I$(top_srcdir)/test \
	-I$(top_builddir)/src \
	-I$(top_srcdir)/test/gtest \
	$(GTEST_CPPFLAGS) \
	$(OPENMP_CFLAGS)

gtest_LDFLAGS  = $(GTEST_LDFLAGS) -no-install -Wl,-dynamic-list-data
gtest_CFLAGS   = $(BASE_CFLAGS)
gtest_CXXFLAGS = \
	$(BASE_CXXFLAGS) $(GTEST_CXXFLAGS) \
	-DGTEST_UCM_HOOK_LIB_DIR="\"${abs_builddir}/ucm/test_dlopen/.libs\""

gtest_SOURCES = \
	common/gtest-all.cc \
	common/main.cc \
	common/test_gtest_cmn.cc \
	common/mem_buffer.cc \
	common/test_helpers.cc \
	common/test_obj_size.cc \
	common/test_watchdog.cc \
	common/test_perf.cc \
	common/test.cc \
	\
	ucm/malloc_hook.cc \
	\
	uct/test_amo.cc \
	uct/test_amo_add_xor.cc \
	uct/test_amo_and_or.cc \
	uct/test_amo_cswap.cc \
	uct/test_amo_fadd_fxor.cc \
	uct/test_amo_fand_for.cc \
	uct/test_amo_swap.cc \
	uct/test_event.cc \
	uct/test_fence.cc \
	uct/test_flush.cc \
	uct/test_many2one_am.cc \
	uct/test_md.cc \
	uct/test_mm.cc \
	uct/test_mem.cc \
	uct/test_p2p_am.cc \
	uct/test_p2p_err.cc \
	uct/test_p2p_mix.cc \
	uct/test_p2p_rma.cc \
	uct/test_pending.cc \
	uct/test_progress.cc \
	uct/test_uct_ep.cc \
	uct/test_uct_perf.cc \
	uct/test_zcopy_comp.cc \
	uct/uct_p2p_test.cc \
	uct/uct_test.cc \
	uct/test_stats.cc \
	ucs/test_event_set.cc \
	ucs/test_stats_filter.cc \
	uct/test_peer_failure.cc \
	uct/test_tag.cc \
	uct/tcp/test_tcp.cc \
	\
	ucp/test_ucp_am.cc \
	ucp/test_ucp_stream.cc \
	ucp/test_ucp_peer_failure.cc \
	ucp/test_ucp_atomic.cc \
	ucp/test_ucp_dt.cc \
	ucp/test_ucp_memheap.cc \
	ucp/test_ucp_mmap.cc \
	ucp/test_ucp_mem_type.cc \
	ucp/test_ucp_perf.cc \
	ucp/test_ucp_proto.cc \
	ucp/test_ucp_rma.cc \
	ucp/test_ucp_rma_mt.cc \
	ucp/test_ucp_tag_cancel.cc \
	ucp/test_ucp_tag_match.cc \
	ucp/test_ucp_tag_offload.cc \
	ucp/test_ucp_tag_mt.cc \
	ucp/test_ucp_tag_perf.cc \
	ucp/test_ucp_tag_probe.cc \
	ucp/test_ucp_tag_xfer.cc \
	ucp/test_ucp_tag_mem_type.cc \
	ucp/test_ucp_tag.cc \
	ucp/test_ucp_context.cc \
	ucp/test_ucp_worker.cc \
	ucp/test_ucp_wireup.cc \
	ucp/test_ucp_wakeup.cc \
	ucp/test_ucp_fence.cc \
	ucp/test_ucp_sockaddr.cc \
	ucp/ucp_test.cc \
	ucp/ucp_datatype.cc \
	\
	ucs/test_algorithm.cc \
	ucs/test_arbiter.cc \
	ucs/test_async.cc \
	ucs/test_callbackq.cc \
	ucs/test_class.cc \
	ucs/test_config.cc \
	ucs/test_conn_match.cc \
	ucs/test_datatype.cc \
	ucs/test_debug.cc \
	ucs/test_memtrack.cc \
	ucs/test_math.cc \
	ucs/test_mpmc.cc \
	ucs/test_mpool.cc \
	ucs/test_pgtable.cc \
	ucs/test_profile.cc \
	ucs/test_rcache.cc \
	ucs/test_memtype_cache.cc \
	ucs/test_stats.cc \
	ucs/test_strided_alloc.cc \
	ucs/test_string.cc \
	ucs/test_sys.cc \
	ucs/test_topo.cc \
	ucs/test_sock.cc \
	ucs/test_time.cc \
	ucs/test_twheel.cc \
	ucs/test_frag_list.cc \
	ucs/test_type.cc \
	ucs/test_log.cc \
	ucs/test_iov.cc \
	ucs/arch/test_x86_64.cc

if HAVE_IB
gtest_SOURCES += \
	uct/ib/test_ib.cc \
	uct/ib/test_ib_md.cc \
	uct/ib/test_cq_moderation.cc \
	uct/ib/test_ib_xfer.cc \
	uct/ib/test_ib_pkey.cc \
	uct/ib/test_ib_event.cc
gtest_CPPFLAGS += \
	$(IBVERBS_CPPFLAGS)
gtest_LDADD += \
	$(IBVERBS_LDFLAGS) \
	$(top_builddir)/src/uct/ib/libuct_ib.la
if HAVE_DEVX
gtest_SOURCES += \
	uct/ib/test_devx.cc
endif
if HAVE_TL_UD
gtest_SOURCES += \
	uct/ib/ud_base.cc \
	uct/ib/test_ud.cc \
	uct/ib/test_ud_timer.cc \
	uct/ib/test_ud_pending.cc \
	uct/ib/test_ud_ds.cc
endif
if HAVE_TL_RC
gtest_SOURCES += \
	uct/ib/test_rc.cc
endif	
if HAVE_TL_DC
gtest_SOURCES += \
	uct/ib/test_dc.cc
endif
if HAVE_RDMACM
gtest_SOURCES += \
	uct/ib/test_sockaddr.cc
endif
endif # HAVE_IB

if HAVE_CUDA
gtest_SOURCES += \
	ucm/cuda_hooks.cc
gtest_CPPFLAGS += \
	$(CUDA_CPPFLAGS)
gtest_LDADD += \
	$(CUDA_LDFLAGS) \
	$(top_builddir)/src/uct/cuda/libuct_cuda.la
endif

if HAVE_HIP
if HAVE_GNUXX11
gtest_SOURCES += \
	ucm/rocm_hooks.cc
gtest_CPPFLAGS += \
	$(HIP_CPPFLAGS)
gtest_CXXFLAGS += \
	$(HIP_CXXFLAGS)
gtest_LDADD += \
	$(HIP_LDFLAGS) \
	$(HIP_LIBS) \
	$(top_builddir)/src/uct/rocm/libuct_rocm.la
endif
endif

noinst_HEADERS = \
	common/gtest.h \
	common/mem_buffer.h \
	common/test.h \
	common/test_helpers.h \
	common/test_perf.h \
	common/tap.h \
	\
	uct/ib/test_rc.h \
	uct/ib/ud_base.h \
	uct/ib/test_ib.h \
	uct/test_amo.h \
	uct/test_p2p_mix.h \
	uct/test_p2p_rma.h \
	uct/uct_p2p_test.h \
	uct/uct_test.h \
	uct/test_md.h \
	uct/test_peer_failure.h \
	\
	ucp/test_ucp_memheap.h \
	ucp/test_ucp_tag.h \
	ucp/ucp_test.h \
	ucp/ucp_datatype.h

.PHONY: test test gdb valgrind fix_rpath ucx


all-local: gtest

ucx:
	$(MAKE) -C $(top_builddir)


help:
	@echo
	@echo "Targets:"
	@echo "  list          : List unit tests."
	@echo "  test          : Run unit tests."
	@echo "  test_gdb      : Run unit tests with GDB."
	@echo "  test_valgrind : Run unit tests with Valgrind."
	@echo
	@echo "Environment variables:"
	@echo "  GTEST_FILTER        : Unit tests filter (\"$(GTEST_FILTER)\")"
	@echo "  GTEST_EXTRA_ARGS    : Additional arguments for gtest (\"$(GTEST_EXTRA_ARGS)\")"
	@echo "  LAUNCHER            : Custom launcher for gtest executable (\"$(LAUNCHER)\")"
	@echo "  VALGRIND_EXTRA_ARGS : Additional arguments for Valgrind (\"$(VALGRIND_EXTRA_ARGS)\")"
	@echo

#
# List unit tests
#
list: gtest
	$(abs_builddir)/gtest --gtest_list_tests $(GTEST_ARGS)

#
# Run unit tests
#
test: ucx gtest
	@rm -f core.*
	$(LAUNCHER) stdbuf -e0 -o0 $(abs_builddir)/gtest $(GTEST_ARGS)

#
# Run unit tests with GDB
#
test_gdb: ucx gtest
	echo -e 'r\ninit-if-undefined $$_exitcode=-1\nif $$_exitcode>=0\n\tq\nend' > .gdbcommands
	$(LAUNCHER) env UCX_HANDLE_ERRORS=none \
		gdb -x .gdbcommands --args $(GDB_ARGS) \
			$(abs_builddir)/gtest $(GTEST_ARGS)

#
# Run unit tests with valgrind
#
test_valgrind: ucx gtest
	$(LAUNCHER) env LD_LIBRARY_PATH="$(VALGRIND_LIBPATH):${LD_LIBRARY_PATH}" \
	stdbuf -e0 -o0 valgrind $(VALGRIND_ARGS) $(abs_builddir)/gtest $(GTEST_ARGS)
endif
