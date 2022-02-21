/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_DEBUG_INT_H_
#define UCS_DEBUG_INT_H_

#include <ucs/datastruct/list.h>
#include <ucs/type/status.h>
#include <ucs/config/types.h>
#include <ucs/debug/debug.h>
#include <stdio.h>


/**
 * Information about an address in the code.
 */
typedef struct ucs_debug_address_info {
    struct {
        char           path[512];          /* Binary file path */
        unsigned long  base;               /* Binary file load base */
    } file;
    char               function[128];      /* Function name */
    char               source_file[512];   /* Source file path */
    unsigned           line_number;        /* Line number */
} ucs_debug_address_info_t;


typedef struct backtrace *backtrace_h;
typedef struct backtrace_line *backtrace_line_h;

extern const char *ucs_state_detail_level_names[];
extern const char *ucs_signal_names[];


/**
 * Initialize UCS debugging subsystem.
 */
void ucs_debug_init();


/**
 * Cleanup UCS debugging subsystem.
 */
void ucs_debug_cleanup(int on_error);

/**
 * Disable signal handling in UCS for all signals
 * that was set in ucs_global_opts.error_signals.
 * Previous signal handlers are set.
 */
void ucs_debug_disable_signals();
/**
 * Get information about an address in the code of the current program.
 * @param address   Address to look up.
 * @param info      Filled with information about the given address. Source file
 *                  and line number are filled only if the binary file was compiled
 *                  with debug information, and UCS was configured with detailed
 *                  backtrace enabled.
 * @return UCS_ERR_NO_ELEM if the address is not found, UCS_OK otherwise.
 */
ucs_status_t ucs_debug_lookup_address(void *address, ucs_debug_address_info_t *info);


/**
 * Create a backtrace from the calling location.
 *
 * @param bckt          Backtrace object.
 * @param strip         How many frames to strip.
*/
ucs_status_t ucs_debug_backtrace_create(backtrace_h *bckt, int strip);


/**
 * Destroy a backtrace and free all memory.
 *
 * @param bckt          Backtrace object.
 */
void ucs_debug_backtrace_destroy(backtrace_h bckt);


/**
 * Walk to the next backtrace line information.
 *
 * @param bckt          Backtrace object.
 * @param line          Filled with backtrace frame info.
 *
 * NOTE: the line remains valid as long as the backtrace object is not destroyed.
 */
int ucs_debug_backtrace_next(backtrace_h bckt, backtrace_line_h *line);


/**
 * Print backtrace line to string buffer.
 *
 * @param buffer         Target buffer to print to.
 * @param maxlen         Size of target buffer.
 * @param frame_num      Frame number
 * @param line           Backtrace line to print
 */
void ucs_debug_print_backtrace_line(char *buffer, size_t maxlen,
                                    int frame_num,
                                    backtrace_line_h line);

/**
 * Print backtrace to an output stream.
 *
 * @param stream         Stream to print to.
 * @param strip          How many frames to strip.
 */
void ucs_debug_print_backtrace(FILE *stream, int strip);


/**
 * Called when UCS detects a fatal error and provides means to debug the current
 * state of UCS.
 */
void ucs_handle_error(const char *message);


/**
 * @return Name of a symbol which begins in the given address, or NULL if
 * not found.
 */
const char *ucs_debug_get_symbol_name(void *address);


/**
 * Check if signal should be processed by UCX as error.
 *
 * @param signum         Signal number to check.
 *
 * @return 1 if signal should be processes by UCX, 0 if passed to system.
 */
int ucs_debug_is_error_signal(int signum);

#endif
