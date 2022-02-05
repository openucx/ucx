/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "global_opts.h"

#include <ucs/config/parser.h>
#include <ucs/profile/profile.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/string.h>
#include <ucs/vfs/base/vfs_obj.h>
#include <sys/signal.h>


ucs_global_opts_t ucs_global_opts = {
    .log_component         = {UCS_LOG_LEVEL_WARN, "UCX", "*"},
    .log_print_enable      = 0,
    .log_file              = "",
    .log_file_size         = SIZE_MAX,
    .log_file_rotate       = 0,
    .log_buffer_size       = 1024,
    .log_data_size         = 0,
    .mpool_fifo            = 0,
    .handle_errors         = UCS_BIT(UCS_HANDLE_ERROR_BACKTRACE),
    .error_signals         = { NULL, 0 },
    .error_mail_to         = "",
    .error_mail_footer     = "",
    .gdb_command           = "gdb",
    .debug_signo           = SIGHUP,
    .log_level_trigger     = UCS_LOG_LEVEL_FATAL,
    .warn_unused_env_vars  = 1,
    .enable_memtype_cache  = UCS_TRY,
    .async_max_events      = 64,
    .async_signo           = SIGALRM,
    .stats_dest            = "",
    .tuning_path           = "",
    .memtrack_dest         = "",
    .memtrack_limit        = UCS_MEMUNITS_INF,
    .stats_trigger         = "exit",
    .profile_mode          = 0,
    .profile_file          = "",
    .stats_filter          = { NULL, 0 },
    .stats_format          = UCS_STATS_FULL,
    .topo_prio             = { NULL, 0 },
    .vfs_enable            = 1,
    .vfs_thread_affinity   = 0,
    .rcache_check_pfn      = 0,
    .module_dir            = UCX_MODULE_DIR, /* defined in Makefile.am */
    .module_log_level      = UCS_LOG_LEVEL_TRACE,
    .modules               = { {NULL, 0}, UCS_CONFIG_ALLOW_LIST_ALLOW_ALL },
    .arch                  = UCS_ARCH_GLOBAL_OPTS_INITALIZER,
    .rcache_stat_min       = 0,
    .rcache_stat_max       = 0
};

static const char *ucs_handle_error_modes[] = {
    [UCS_HANDLE_ERROR_BACKTRACE] = "bt",
    [UCS_HANDLE_ERROR_FREEZE]    = "freeze",
    [UCS_HANDLE_ERROR_DEBUG]     = "debug",
    [UCS_HANDLE_ERROR_NONE]      = "none",
    [UCS_HANDLE_ERROR_LAST]      = NULL
};


static UCS_CONFIG_DEFINE_ARRAY(signo,
                               sizeof(int),
                               UCS_CONFIG_TYPE_SIGNO);


static ucs_config_field_t ucs_global_opts_table[] = {
 {"LOG_LEVEL", "warn",
  "UCS logging level. Messages with a level higher or equal to the selected "
  "will be printed.\n"
  "Possible values are: fatal, error, warn, info, debug, trace, data, func, poll.",
  ucs_offsetof(ucs_global_opts_t, log_component.log_level),
  UCS_CONFIG_TYPE_LOG_COMP},

 {"LOG_FILE_FILTER", "*",
  "Set a filter for log message according to source file path. See glob (7) for\n"
  "pattern syntax.\n"
  "NOTE: The source file path must fully match the given pattern.",
  ucs_offsetof(ucs_global_opts_t, log_component.file_filter),
               UCS_CONFIG_TYPE_STRING},

 {"LOG_BUFFER", "1024",
  "Buffer size for a single log message.",
  ucs_offsetof(ucs_global_opts_t, log_buffer_size), UCS_CONFIG_TYPE_MEMUNITS},

 {"LOG_DATA_SIZE", "0",
  "How much packet payload to print, at most, in data mode.",
  ucs_offsetof(ucs_global_opts_t, log_data_size), UCS_CONFIG_TYPE_ULONG},

 {"LOG_PRINT_ENABLE", "n",
  "Enable output of ucs_print(). This option is intended for use by the library developers.",
  ucs_offsetof(ucs_global_opts_t, log_print_enable), UCS_CONFIG_TYPE_BOOL},

#if ENABLE_DEBUG_DATA
 {"MPOOL_FIFO", "n",
  "Enable FIFO behavior for memory pool, instead of LIFO. Useful for\n"
  "debugging because object pointers are not recycled.",
  ucs_offsetof(ucs_global_opts_t, mpool_fifo), UCS_CONFIG_TYPE_BOOL},
#endif

 {"HANDLE_ERRORS",
#if ENABLE_DEBUG_DATA
  "bt,freeze",
#else
  "bt",
#endif
  "Error signal handling mode. Either 'none' to disable signal interception,\n"
  "or a combination of:\n"
  " - 'bt'     : Print backtrace\n"
  " - 'freeze' : Freeze and wait for a debugger\n"
  " - 'debug'  : Attach a debugger",
  ucs_offsetof(ucs_global_opts_t, handle_errors),
  UCS_CONFIG_TYPE_BITMAP(ucs_handle_error_modes)},

 {"ERROR_MAIL_TO", "",
  "If non-empty, send mail notification for fatal errors.",
  ucs_offsetof(ucs_global_opts_t, error_mail_to), UCS_CONFIG_TYPE_STRING},

 {"ERROR_MAIL_FOOTER", "",
  "Footer for error report email",
  ucs_offsetof(ucs_global_opts_t, error_mail_footer), UCS_CONFIG_TYPE_STRING},

 {"GDB_COMMAND", "gdb -quiet",
  "If non-empty, attaches a gdb to the process in case of error, using the provided command.",
  ucs_offsetof(ucs_global_opts_t, gdb_command), UCS_CONFIG_TYPE_STRING},

 {"DEBUG_SIGNO", "SIGHUP",
  "Signal number which causes UCS to enter debug mode. Set to 0 to disable.",
  ucs_offsetof(ucs_global_opts_t, debug_signo), UCS_CONFIG_TYPE_SIGNO},

 {"LOG_LEVEL_TRIGGER", "fatal",
  "Log level to trigger error handling.",
  ucs_offsetof(ucs_global_opts_t, log_level_trigger), UCS_CONFIG_TYPE_ENUM(ucs_log_level_names)},

 {UCS_GLOBAL_OPTS_WARN_UNUSED_CONFIG, "yes",
  "Issue warning about UCX_ environment variables which were not used by the\n"
  "configuration parser.",
  ucs_offsetof(ucs_global_opts_t, warn_unused_env_vars), UCS_CONFIG_TYPE_BOOL},

  {"MEMTYPE_CACHE", "try",
   "Enable memory type (cuda/rocm) cache",
   ucs_offsetof(ucs_global_opts_t, enable_memtype_cache), UCS_CONFIG_TYPE_TERNARY},

 {"ASYNC_MAX_EVENTS", "1024", /* TODO remove this; resize mpmc */
  "Maximal number of events which can be handled from one context",
  ucs_offsetof(ucs_global_opts_t, async_max_events), UCS_CONFIG_TYPE_UINT},

 {"ASYNC_SIGNO", "SIGALRM",
  "Signal number used for async signaling.",
  ucs_offsetof(ucs_global_opts_t, async_signo), UCS_CONFIG_TYPE_SIGNO},

 {"MEMTRACK_LIMIT", "inf",
  "Memory limit allocated by memtrack. In case if limit is reached then\n"
  "memtrack report is generated and process is terminated.",
  ucs_offsetof(ucs_global_opts_t, memtrack_limit), UCS_CONFIG_TYPE_MEMUNITS},

 {"RCACHE_CHECK_PFN", "0",
  "Registration cache to check that the physical pages frame number of a found\n"
  "memory region were not changed since the time the region was registered.\n"
  "Number of pages to check, 0 - disable checking.",
  ucs_offsetof(ucs_global_opts_t, rcache_check_pfn), UCS_CONFIG_TYPE_UINT},

 {"MODULE_DIR", UCX_MODULE_DIR,
  "Directory to search for loadable modules",
  ucs_offsetof(ucs_global_opts_t, module_dir), UCS_CONFIG_TYPE_STRING},

 {"MODULE_LOG_LEVEL", "trace",
  "Logging level for module loader",
  ucs_offsetof(ucs_global_opts_t, module_log_level), UCS_CONFIG_TYPE_ENUM(ucs_log_level_names)},

 {"MODULES", "all",
  "Comma-separated list of glob patterns specifying which module to load.\n"
  "The order is not meaningful. For example:\n"
  " *     - load all modules\n"
  " ^cu*  - do not load modules that begin with 'cu'",
  ucs_offsetof(ucs_global_opts_t, modules), UCS_CONFIG_TYPE_ALLOW_LIST},

 {"TOPO_PRIO", "sysfs,default",
  "Comma-separated list of methods of detecting system topology.\n"
  "The list order decides the priority of methods used.",
  ucs_offsetof(ucs_global_opts_t, topo_prio), UCS_CONFIG_TYPE_STRING_ARRAY},

 {NULL}
};

UCS_CONFIG_DECLARE_TABLE(ucs_global_opts_table, "UCS global", NULL,
                         ucs_global_opts_t)


static ucs_config_field_t ucs_global_opts_read_only_table[] = {
 {"LOG_FILE", "",
  "If not empty, UCS will print log messages to the specified file instead of stdout.\n"
  "The following substitutions are performed on this string:\n"
  "  %p - Replaced with process ID\n"
  "  %h - Replaced with host name",
  ucs_offsetof(ucs_global_opts_t, log_file),
  UCS_CONFIG_TYPE_STRING},

 {"LOG_FILE_SIZE", "inf",
  "The maximal size of log file. The maximal log file size has to be >= LOG_BUFFER.",
  ucs_offsetof(ucs_global_opts_t, log_file_size), UCS_CONFIG_TYPE_MEMUNITS},

 {"LOG_FILE_ROTATE", "0",
  "The maximal number of backup log files that could be created to save logs\n"
  "after the previous ones (if any) are completely filled. The value has to be\n"
  "less than the maximal signed integer value.",
  ucs_offsetof(ucs_global_opts_t, log_file_rotate), UCS_CONFIG_TYPE_UINT},

 {"ERROR_SIGNALS", "SIGILL,SIGSEGV,SIGBUS,SIGFPE",
  "Signals which are considered an error indication and trigger error handling.",
  ucs_offsetof(ucs_global_opts_t, error_signals), UCS_CONFIG_TYPE_ARRAY(signo)},

 {"VFS_ENABLE", "y",
  "Enable virtual monitoring filesystem",
  ucs_offsetof(ucs_global_opts_t, vfs_enable), UCS_CONFIG_TYPE_BOOL},

 {"VFS_THREAD_AFFINITY", "n",
  "Enable inheriting main process affinity for virtual monitoring filesystem\n"
  "service thread. Setting this value to 'n' will allow the service thread to\n"
  "run on any CPU core.",
  ucs_offsetof(ucs_global_opts_t, vfs_thread_affinity),
  UCS_CONFIG_TYPE_BOOL},

#ifdef ENABLE_STATS
 {"STATS_DEST", "",
  "Destination to send statistics to. If the value is empty, statistics are\n"
  "not reported. Possible values are:\n"
  "  udp:<host>[:<port>]   - send over UDP to the given host:port.\n"
  "  stdout                - print to standard output.\n"
  "  stderr                - print to standard error.\n"
  "  file:<filename>[:bin] - save to a file (%h: host, %p: pid, %c: cpu, %t: time, %u: user, %e: exe)",
  ucs_offsetof(ucs_global_opts_t, stats_dest), UCS_CONFIG_TYPE_STRING},

 {"STATS_TRIGGER", "exit",
  "Trigger to dump statistics:\n"
  "  exit              - dump just before program exits.\n"
  "  signal:<signo>    - dump when process is signaled.\n"
  "  timer:<interval>  - dump in specified intervals (in seconds).",
  ucs_offsetof(ucs_global_opts_t, stats_trigger), UCS_CONFIG_TYPE_STRING},

 {"STATS_FILTER", "*",
  "Used for filter counters summary.\n"
  "Comma-separated list of glob patterns specifying counters.\n"
  "Statistics summary will contain only the matching counters.\n"
  "The order is not meaningful.\n"
  "Each expression in the list may contain any of the following wildcard:\n"
  "  *     - matches any number of any characters including none.\n"
  "  ?     - matches any single character.\n"
  "  [abc] - matches one character given in the bracket.\n"
  "  [a-z] - matches one character from the range given in the bracket.",
  ucs_offsetof(ucs_global_opts_t, stats_filter), UCS_CONFIG_TYPE_STRING_ARRAY},

 {"STATS_FORMAT", "full",
  "Statistics format parameter:\n"
  "  full    - each counter will be displayed in a separate line \n"
  "  agg     - like full but there will also be an aggregation between similar counters\n"
  "  summary - all counters will be printed in the same line.",
  ucs_offsetof(ucs_global_opts_t, stats_format), UCS_CONFIG_TYPE_ENUM(ucs_stats_formats_names)},
#endif

 {"MEMTRACK_DEST", "",
  "Destination to output memory tracking report to. If the value is empty,\n"
  "results are not reported. Possible values are:\n"
  "  file:<filename>   - save to a file (%h: host, %p: pid, %c: cpu, %t: time, %u: user, %e: exe)\n"
  "  stdout            - print to standard output.\n"
  "  stderr            - print to standard error.",
  ucs_offsetof(ucs_global_opts_t, memtrack_dest), UCS_CONFIG_TYPE_STRING},

 {"PROFILE_MODE", "",
  "Profile collection modes. If none is specified, profiling is disabled.\n"
  " - log   - Record all timestamps.\n"
  " - accum - Accumulate measurements per location.",
  ucs_offsetof(ucs_global_opts_t, profile_mode),
  UCS_CONFIG_TYPE_BITMAP(ucs_profile_mode_names)},

 {"PROFILE_FILE", "ucx_%h_%p.prof",
  "File name to dump profiling data to.\n"
  "Substitutions: %h: host, %p: pid, %c: cpu, %t: time, %u: user, %e: exe.",
  ucs_offsetof(ucs_global_opts_t, profile_file), UCS_CONFIG_TYPE_STRING},

 {"PROFILE_LOG_SIZE", "4m",
  "Maximal size of profiling log. New records will replace old records.",
  ucs_offsetof(ucs_global_opts_t, profile_log_size), UCS_CONFIG_TYPE_MEMUNITS},

 {"RCACHE_STAT_MIN", "4k",
  "Registration cache minimum region size, for power-of-2 size distribution "
  "statistics.\nStatistics about smaller regions will be attributed to this "
  "specified minimal size.\nRounded up to the next power-of-2 value.",
  ucs_offsetof(ucs_global_opts_t, rcache_stat_min), UCS_CONFIG_TYPE_MEMUNITS},

 {"RCACHE_STAT_MAX", "1m",
  "Registration cache maximum region size, for power-of-2 size distribution "
  "statistics.\nStatistics about larger regions will be attributed to 'max' "
  "bucket.\nRounded up to the next power-of-2 value.",
  ucs_offsetof(ucs_global_opts_t, rcache_stat_max), UCS_CONFIG_TYPE_MEMUNITS},

 {"", "", NULL,
  ucs_offsetof(ucs_global_opts_t, arch),
  UCS_CONFIG_TYPE_TABLE(ucs_arch_global_opts_table)},

 {NULL}
};

UCS_CONFIG_DECLARE_TABLE(ucs_global_opts_read_only_table,
                         "UCS global (runtime read-only)", NULL,
                         ucs_global_opts_t)


ucs_status_t ucs_global_opts_set_value(const char *name, const char *value)
{
    ucs_status_t status = ucs_global_opts_set_value_modifiable(name, value);

    if (status != UCS_ERR_NO_ELEM) {
        return status;
    }

    return ucs_config_parser_set_value(&ucs_global_opts,
                                       ucs_global_opts_read_only_table,
                                       name, value);
}

ucs_status_t ucs_global_opts_set_value_modifiable(const char *name,
                                                  const char *value)
{
    return ucs_config_parser_set_value(&ucs_global_opts,
                                       ucs_global_opts_table, name,
                                       value);
}

ucs_status_t ucs_global_opts_get_value(const char *name, char *value,
                                       size_t max)
{
    ucs_status_t status =
            ucs_config_parser_get_value(&ucs_global_opts,
                                        ucs_global_opts_table, name, value,
                                        max);

    if (status != UCS_ERR_NO_ELEM) {
        return status;
    }

    return ucs_config_parser_get_value(&ucs_global_opts,
                                       ucs_global_opts_read_only_table, name,
                                       value, max);
}

ucs_status_t ucs_global_opts_clone(void *dst)
{
    ucs_status_t status =
            ucs_config_parser_clone_opts(&ucs_global_opts, dst,
                                         ucs_global_opts_table);

    if (status != UCS_OK) {
        return status;
    }

    /* Both modifiable and read-only tables of global options use the common
     * storage for parameters. So, cloning parameters to the same destination
     * is ok due to different offsets of parameter's fields in the storage */
    return ucs_config_parser_clone_opts(&ucs_global_opts, dst,
                                        ucs_global_opts_read_only_table);
}

void ucs_global_opts_release()
{
    ucs_config_parser_release_opts(&ucs_global_opts, ucs_global_opts_table);
    ucs_config_parser_release_opts(&ucs_global_opts,
                                   ucs_global_opts_read_only_table);
}

void ucs_global_opts_print(FILE *stream, ucs_config_print_flags_t print_flags)
{
    ucs_config_parser_print_opts(stream, "Global configuration",
                                 &ucs_global_opts, ucs_global_opts_table, NULL,
                                 UCS_DEFAULT_ENV_PREFIX, print_flags);
    ucs_config_parser_print_opts(stream,
                                 "Global configuration (runtime read-only)",
                                 &ucs_global_opts,
                                 ucs_global_opts_read_only_table, NULL,
                                 UCS_DEFAULT_ENV_PREFIX, print_flags);
}

static void ucs_vfs_read_log_level(void *obj, ucs_string_buffer_t *strb,
                                   void *arg_ptr, uint64_t arg_u64)
{
    ucs_log_level_t ucs_log_level = ucs_global_opts.log_component.log_level;
    ucs_string_buffer_appendf(strb, "%s\n", ucs_log_level_names[ucs_log_level]);
}

static ucs_status_t ucs_vfs_write_log_level(void *obj, const char *buffer,
                                            size_t size, void *arg_ptr,
                                            uint64_t arg_u64)
{
    UCS_STRING_BUFFER_ONSTACK(strb, 32);
    unsigned log_level;

    ucs_string_buffer_appendf(&strb, "%s", buffer);
    ucs_string_buffer_rtrim(&strb, "\n");
    if (!ucs_config_sscanf_enum(ucs_string_buffer_cstr(&strb), &log_level,
                                ucs_log_level_names)) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_global_opts.log_component.log_level = log_level;
    return UCS_OK;
}

void ucs_global_opts_init()
{
    ucs_status_t status;

    UCS_CONFIG_ADD_TABLE(ucs_global_opts_table,  &ucs_config_global_list);
    UCS_CONFIG_ADD_TABLE(ucs_global_opts_read_only_table,
                         &ucs_config_global_list);

    status = ucs_config_parser_fill_opts(&ucs_global_opts,
                                         ucs_global_opts_read_only_table,
                                         UCS_DEFAULT_ENV_PREFIX, NULL, 1);
    if (status != UCS_OK) {
        ucs_fatal("failed to parse global runtime read-only configuration");
    }

    status = ucs_config_parser_fill_opts(&ucs_global_opts,
                                         ucs_global_opts_table,
                                         UCS_DEFAULT_ENV_PREFIX, NULL, 1);
    if (status != UCS_OK) {
        ucs_fatal("failed to parse global configuration");
    }

    /**
     * VFS nodes representing global options are removed in UCS_STATIC_CLEANUP
     * of vfs_obj.c file. Because removing elements using ucs_vfs_obj_remove
     * in UCS_STATIC_CLEANUP in global_opts.c could happen after execution of
     * UCS_STATIC_CLEANUP of vfs_obj.c file.
     */
    ucs_vfs_obj_add_dir(NULL, &ucs_global_opts, "ucs/global_opts");
    ucs_vfs_obj_add_rw_file(&ucs_global_opts, ucs_vfs_read_log_level,
                            ucs_vfs_write_log_level, NULL, 0, "log_level");
}

void ucs_global_opts_cleanup()
{
    UCS_CONFIG_REMOVE_TABLE(ucs_global_opts_read_only_table);
    UCS_CONFIG_REMOVE_TABLE(ucs_global_opts_table);
}
