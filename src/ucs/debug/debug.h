/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_DEBUG_H_
#define UCS_DEBUG_H_

#include <ucs/datastruct/list.h>
#include <ucs/type/status.h>
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


extern const char *ucs_state_detail_level_names[];
extern const char *ucs_signal_names[];


/**
 * Initialize UCS debugging subsystem.
 */
void ucs_debug_init();


/**
 * Cleanup UCS debugging subsystem.
 */
void ucs_debug_cleanup();


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
 * @return Full path to current library.
 */
const char *ucs_debug_get_lib_path();


/**
 * @return UCS library loading address.
 */
unsigned long ucs_debug_get_lib_base_addr();


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
void ucs_handle_error(const char *error_type, const char *message, ...);


/**
 * @return Name of a symbol which begins in the given address, or NULL if
 * not found.
 */
const char *ucs_debug_get_symbol_name(void *address);


#endif
