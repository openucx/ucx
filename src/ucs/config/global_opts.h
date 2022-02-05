/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_CONFIG_H_
#define UCS_CONFIG_H_

#include "types.h"

#include <ucs/stats/stats_fwd.h>
#include <ucs/type/status.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/arch/global_opts.h>
#include <stddef.h>
#include <stdio.h>

BEGIN_C_DECLS

/** @file global_opts.h */

#define UCS_GLOBAL_OPTS_WARN_UNUSED_CONFIG    "WARN_UNUSED_ENV_VARS"

/**
 * UCS global options.
 */
typedef struct {

    /* Log level above which log messages will be printed for default component */
    ucs_log_component_config_t log_component;

    /* Log file */
    char                       *log_file;

    /* Maximal log file size */
    size_t                     log_file_size;

    /* Maximal backup log files count that could be created by log infrastructure */
    unsigned                   log_file_rotate;

    /* Size of log buffer for one message */
    size_t                     log_buffer_size;

    /* Maximal amount of packet data to print per packet */
    size_t                     log_data_size;

    /* Enable ucs_print() output */
    int                        log_print_enable;

    /* Enable FIFO behavior for memory pool, instead of LIFO. Useful for
     * debugging because object pointers are not recycled. */
    int                        mpool_fifo;

    /* Handle errors mode */
    unsigned                   handle_errors;

    /* Error signals */
    UCS_CONFIG_ARRAY_FIELD(int, signals) error_signals;

    /* If not empty, send mail notifications to that address in case of error */
    char                       *error_mail_to;

    /* Footer for error report mail notification */
    char                       *error_mail_footer;

    /* If not NULL, attach gdb to the process in case of error */
    char                       *gdb_command;

    /* Signal number which causes to enter debug mode */
    unsigned                   debug_signo;

    /* Log level to trigger error handling */
    ucs_log_level_t            log_level_trigger;

    /* Issue warning about UCX_ env vars which were not used by config parser */
    int                        warn_unused_env_vars;

    /* Max. events per context, will be removed in the future */
    unsigned                   async_max_events;

    /** Memtype cache */
    ucs_ternary_auto_value_t   enable_memtype_cache;

    /* Destination for statistics: udp:host:port / file:path / stdout
     */
    char                       *stats_dest;

    /* Trigger to dump statistics */
    char                       *stats_trigger;

    /* Named pipe file path for tuning.
     */
    char                       *tuning_path;

    /* Number of performance stall loops to perform */
    size_t                     perf_stall_loops;

    /* Signal number used by async handler (for signal mode) */
    unsigned                   async_signo;

    /* Destination for detailed memory tracking results: none / stdout / stderr
     */
    char                       *memtrack_dest;

    /* Memory limit handled by memtrack to abort application */
    size_t                     memtrack_limit;

    /* Profiling mode */
    unsigned                   profile_mode;

    /* Profiling output file name */
    char                       *profile_file;

    /* Limit for profiling log size */
    size_t                     profile_log_size;

    /* Counters to be included in statistics summary */
    ucs_config_names_array_t   stats_filter;

    /* statistics format options */
    ucs_stats_formats_t        stats_format;

    /* Topology detection modules to use */
    ucs_config_names_array_t   topo_prio;

    /* Enable VFS monitoring */
    int                        vfs_enable;

    /* registration cache checks if physical pages are not moved */
    unsigned                   rcache_check_pfn;

    /* directory for loadable modules */
    char                       *module_dir;

    /* log level for module loader code */
    ucs_log_level_t            module_log_level;

    /* which modules to load */
    ucs_config_allow_list_t    modules;

    /* arch-specific global options */
    ucs_arch_global_opts_t     arch;

    /* Enable affinity for virtual monitoring filesystem service thread */
    int                        vfs_thread_affinity;

    /* Boundary thresholds for the size distribution of registered cache
       regions. Intermediate thresholds are power-of-2 numbers between these
       values. */
    size_t                     rcache_stat_min;
    size_t                     rcache_stat_max;
} ucs_global_opts_t;


extern ucs_global_opts_t ucs_global_opts;

void ucs_global_opts_init();
void ucs_global_opts_cleanup();
ucs_status_t ucs_global_opts_set_value(const char *name, const char *value);
ucs_status_t ucs_global_opts_set_value_modifiable(const char *name,
                                                  const char *value);
ucs_status_t ucs_global_opts_get_value(const char *name, char *value,
                                       size_t max);
ucs_status_t ucs_global_opts_clone(void *dst);
void ucs_global_opts_release();
void ucs_global_opts_print(FILE *stream, ucs_config_print_flags_t print_flags);

END_C_DECLS

#endif
