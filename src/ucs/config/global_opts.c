/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "global_opts.h"

#include <ucs/config/parser.h>
#include <ucs/debug/profile.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <sys/signal.h>


ucs_global_opts_t ucs_global_opts = {
    .log_level             = UCS_LOG_LEVEL_WARN,
    .log_print_enable      = 0,
    .log_file              = "",
    .log_buffer_size       = 1024,
    .log_data_size         = 0,
    .mpool_fifo            = 0,
    .handle_errors         = UCS_BIT(UCS_HANDLE_ERROR_BACKTRACE),
    .error_signals         = { NULL, 0 },
    .error_mail_to         = "",
    .error_mail_footer     = "",
    .gdb_command           = "gdb",
    .debug_signo           = SIGHUP,
    .async_max_events      = 64,
    .async_signo           = SIGALRM,
    .stats_dest            = "",
    .tuning_path           = "",
    .memtrack_dest         = "",
    .stats_trigger         = "exit",
    .profile_mode          = 0,
    .profile_file          = "",
    .stats_filter          = { NULL, 0 },
    .stats_format          = UCS_STATS_FULL,
};

static const char *ucs_handle_error_modes[] = {
    [UCS_HANDLE_ERROR_BACKTRACE] = "bt",
    [UCS_HANDLE_ERROR_FREEZE]    = "freeze",
    [UCS_HANDLE_ERROR_DEBUG]     = "debug",
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
  ucs_offsetof(ucs_global_opts_t, log_level), UCS_CONFIG_TYPE_ENUM(ucs_log_level_names)},

 {"LOG_FILE", "",
  "If not empty, UCS will print log messages to the specified file instead of stdout.\n"
  "The following substitutions are performed on this string:\n"
  "  %p - Replaced with process ID\n"
  "  %h - Replaced with host name\n",
  ucs_offsetof(ucs_global_opts_t, log_file),
  UCS_CONFIG_TYPE_STRING},

 {"LOG_BUFFER", "1024",
  "Buffer size for a single log message.",
  ucs_offsetof(ucs_global_opts_t, log_buffer_size), UCS_CONFIG_TYPE_MEMUNITS},

 {"LOG_DATA_SIZE", "0",
  "How much packet payload to print, at most, in data mode.",
  ucs_offsetof(ucs_global_opts_t, log_data_size), UCS_CONFIG_TYPE_ULONG},

 {"LOG_PRINT_ENABLE", "n",
  "Enable output of ucs_print(). This option is intended for use by the library developers.\n",
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
  "Error handling mode. A combination of: 'bt' (print backtrace),\n"
  "'freeze' (freeze and wait for a debugger), 'debug' (attach debugger)",
  ucs_offsetof(ucs_global_opts_t, handle_errors),
  UCS_CONFIG_TYPE_BITMAP(ucs_handle_error_modes)},

 {"ERROR_SIGNALS", "SIGILL,SIGSEGV,SIGBUS,SIGFPE",
  "Signals which are considered an error indication and trigger error handling.",
  ucs_offsetof(ucs_global_opts_t, error_signals), UCS_CONFIG_TYPE_ARRAY(signo)},

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

 {"ASYNC_MAX_EVENTS", "1024", /* TODO remove this; resize mpmc */
  "Maximal number of events which can be handled from one context",
  ucs_offsetof(ucs_global_opts_t, async_max_events), UCS_CONFIG_TYPE_UINT},

 {"ASYNC_SIGNO", "SIGALRM",
  "Signal number used for async signaling.",
  ucs_offsetof(ucs_global_opts_t, async_signo), UCS_CONFIG_TYPE_SIGNO},

#if ENABLE_STATS
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

#if ENABLE_MEMTRACK
 {"MEMTRACK_DEST", "",
  "Destination to output memory tracking report to. If the value is empty,\n"
  "results are not reported. Possible values are:\n"
  "  file:<filename>   - save to a file (%h: host, %p: pid, %c: cpu, %t: time, %u: user, %e: exe)\n"
  "  stdout            - print to standard output.\n"
  "  stderr            - print to standard error.\n",
  ucs_offsetof(ucs_global_opts_t, memtrack_dest), UCS_CONFIG_TYPE_STRING},
#endif

#if HAVE_PROFILING
  {"PROFILE_MODE", "",
   "Profile collection modes. If none is specified, profiling is disabled.\n"
   " - log   - Record all timestamps.\n"
   " - accum - Accumulate measurements per location.\n",
   ucs_offsetof(ucs_global_opts_t, profile_mode),
   UCS_CONFIG_TYPE_BITMAP(ucs_profile_mode_names)},

  {"PROFILE_FILE", "",
   "File name to dump profiling data to.\n"
   "Substitutions: %h: host, %p: pid, %c: cpu, %t: time, %u: user, %e: exe.\n",
   ucs_offsetof(ucs_global_opts_t, profile_file), UCS_CONFIG_TYPE_STRING},

  {"PROFILE_LOG_SIZE", "4m",
   "Maximal size of profiling log. New records will replace old records.",
   ucs_offsetof(ucs_global_opts_t, profile_log_size), UCS_CONFIG_TYPE_MEMUNITS},
#endif

 {NULL}
};

void ucs_global_opts_init()
{
    ucs_status_t status;

    status = ucs_config_parser_fill_opts(&ucs_global_opts, ucs_global_opts_table,
                                         NULL, NULL, 1);
    if (status != UCS_OK) {
        ucs_fatal("failed to parse global configuration - aborting");
    }
}

ucs_status_t ucs_global_opts_set_value(const char *name, const char *value)
{
    return ucs_config_parser_set_value(&ucs_global_opts, ucs_global_opts_table,
                                       name, value);
}

ucs_status_t ucs_global_opts_get_value(const char *name, char *value, size_t max)
{
    return ucs_config_parser_get_value(&ucs_global_opts, ucs_global_opts_table,
                                       name, value, max);
}

ucs_status_t ucs_global_opts_clone(void *dst)
{
    return ucs_config_parser_clone_opts(&ucs_global_opts, dst, ucs_global_opts_table);
}

void ucs_global_opts_release()
{
    return ucs_config_parser_release_opts(&ucs_global_opts, ucs_global_opts_table);
}

void ucs_global_opts_print(FILE *stream, ucs_config_print_flags_t print_flags)
{
    ucs_config_parser_print_opts(stream, "Global configuration", &ucs_global_opts,
                                 ucs_global_opts_table, NULL, print_flags);
}
