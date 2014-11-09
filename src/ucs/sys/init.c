/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <ucs/sys/compiler.h>
#include <ucs/debug/debug.h>
#include <ucs/debug/instrument.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/stats/stats.h>

static void UCS_F_CTOR ucs_init()
{
    ucs_log_early_init(); /* Must be called before all others */
    ucs_global_opts_init();
    ucs_log_init();
    ucs_stats_init();
    ucs_memtrack_init();
    ucs_debug_init();
    ucs_instrument_init();
}

static void UCS_F_DTOR ucs_cleanup(void)
{
    ucs_instrument_cleanup();
    ucs_debug_cleanup();
    ucs_memtrack_cleanup();
    ucs_stats_cleanup();
    ucs_log_cleanup();
}
