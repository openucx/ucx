/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "global_opts.h"

#include <ucs/debug/log.h>
#include <sys/signal.h>


ucs_global_opts_t ucs_global_opts = {
    .log_level             = UCS_LOG_LEVEL_WARN,
    .log_file              = "",
    .log_buffer_size       = 1024,
    .log_data_size         = 0,
    .handle_errors         = UCS_HANDLE_ERROR_BACKTRACE,
    .error_signals         = { NULL, 0 },
    .gdb_command           = "gdb",
    .debug_signo           = SIGHUP,
    .async_interval        = 0.05,
    .async_signo           = SIGALRM,
    .stats_dest            = "",
    .tuning_path           = "",
    .instrument_file       = "",
    .instrument_max_size   = 1048576,
    .memtrack_dest         = "",
#if ENABLE_STATS
    .stats_dest            = "",
    .stats_trigger         = "exit",
#endif
#if ENABLE_MEMTRACK
    .memtrack_dest         = "",
#endif
#if HAVE_INSTRUMENTATION
    .instrument_file       = "",
    .instrument_max_size   = 1048576,
#endif
};

static const char *handle_error_modes[] = {
    [UCS_HANDLE_ERROR_NONE]      = "none",
    [UCS_HANDLE_ERROR_BACKTRACE] = "bt",
    [UCS_HANDLE_ERROR_FREEZE]    = "freeze",
    [UCS_HANDLE_ERROR_DEBUG]     = "debug",
    [UCS_HANDLE_ERROR_LAST]      = NULL
};

static UCS_CONFIG_DEFINE_ARRAY(signo,
                               sizeof(int),
                               UCS_CONFIG_TYPE_SIGNO);

ucs_config_field_t ucs_global_opts_table[] = {
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

 {"HANDLE_ERRORS", "bt",
  "Error handling mode. Possible values are: 'none' (no error handling), 'bt' (print\n"
  "backtrace), 'freeze' (freeze and wait for a debugger), 'debug' (attach debugger)",
  ucs_offsetof(ucs_global_opts_t, handle_errors), UCS_CONFIG_TYPE_ENUM(handle_error_modes)},

 {"ERROR_SIGNALS", "SIGILL,SIGSEGV,SIGBUS,SIGFPE",
  "Signals which are considered an error indication and trigger error handling.",
  ucs_offsetof(ucs_global_opts_t, error_signals), UCS_CONFIG_TYPE_ARRAY(signo)},

 {"GDB_COMMAND", "gdb",
  "If non-empty, attaches a gdb to the process in case of error, using the provided command.",
  ucs_offsetof(ucs_global_opts_t, gdb_command), UCS_CONFIG_TYPE_STRING},

 {"DEBUG_SIGNO", "SIGHUP",
  "Signal number which causes UCS to enter debug mode. Set to 0 to disable.",
  ucs_offsetof(ucs_global_opts_t, debug_signo), UCS_CONFIG_TYPE_SIGNO},

 {"ASYNC_INTERVAL", "50ms",
  "Interval of asynchronous progress. Lower values may make the network\n"
  "more responsive, in the cost of higher CPU load.",
  ucs_offsetof(ucs_global_opts_t, async_interval), UCS_CONFIG_TYPE_TIME},

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

#if HAVE_INSTRUMENTATION
 {"INSTRUMENT", "",
  "File name to dump instrumentation records to.\n"
  "Substitutions: %h: host, %p: pid, %c: cpu, %t: time, %u: user, %e: exe.\n",
  ucs_offsetof(ucs_global_opts_t, instrument_file),
  UCS_CONFIG_TYPE_STRING},

 {"INSTRUMENT_SIZE", "1048576",
  "Maximal size of instrumentation data. New records will replace old records.",
  ucs_offsetof(ucs_global_opts_t, instrument_max_size),
  UCS_CONFIG_TYPE_MEMUNITS},
#endif

 {NULL}
};

void ucs_global_opts_init()
{
    ucs_status_t status;

    status = ucs_config_parser_fill_opts(&ucs_global_opts, ucs_global_opts_table,
                                         "UCS_", NULL, 1);
    if (status != UCS_OK) {
        ucs_fatal("failed to parse global configuration - aborting");
    }
}
