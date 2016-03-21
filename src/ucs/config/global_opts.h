/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_CONFIG_H_
#define UCS_CONFIG_H_

#include "types.h"

#include <ucs/type/status.h>
#include <stddef.h>
#include <stdio.h>



/**
 * UCS global options.
 */
typedef struct {

    /* Log level above which log messages will be printed */
    ucs_log_level_t          log_level;

    /* Log file */
    char                     *log_file;

    /* Size of log buffer for one message */
    size_t                   log_buffer_size;

    /* Maximal amount of packet data to print per packet */
    size_t                   log_data_size;

    /* Enable FIFO behavior for memory pool, instead of LIFO. Useful for
     * debugging because object pointers are not recycled. */
    int                      mpool_fifo;

    /* Handle errors mode */
    ucs_handle_error_t       handle_errors;

    /* Error signals */
    UCS_CONFIG_ARRAY_FIELD(int, signals) error_signals;

    /* If not NULL, attach gdb to the process in case of error */
    char                     *gdb_command;

    /* Signal number which causes to enter debug mode */
    unsigned                 debug_signo;

    /* File name to dump instrumentation records to */
    char                     *instrument_file;

    /* Types of instrumentation to be recorded for this run */
    unsigned                 instrument_types;

    /* Limit for instrumentation data size */
    size_t                   instrument_max_size;

    /* Max. events per context, will be removed in the future */
    unsigned                 async_max_events;

    /* Destination for statistics: udp:host:port / file:path / stdout
     */
    char                     *stats_dest;

    /* Trigger to dump statistics */
    char                     *stats_trigger;

    /* Named pipe file path for tuning.
     */
    char                     *tuning_path;

    /* Number of performance stall loops to perform */
    size_t                   perf_stall_loops;

    /* Signal number used by async handler (for signal mode) */
    unsigned                 async_signo;

    /* Destination for detailed memory tracking results: none / stdout / stderr
     */
    char                     *memtrack_dest;

} ucs_global_opts_t;


extern ucs_global_opts_t ucs_global_opts;

void ucs_global_opts_init();
ucs_status_t ucs_global_opts_set_value(const char *name, const char *value);
ucs_status_t ucs_global_opts_clone(void *dst);
void ucs_global_opts_release();
void ucs_global_opts_print(FILE *stream, ucs_config_print_flags_t print_flags);

#endif
