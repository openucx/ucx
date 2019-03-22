/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_PROFILE_DEFS_H_
#define UCS_PROFILE_DEFS_H_

#include <ucs/config/global_opts.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/time/time_def.h>


BEGIN_C_DECLS

/** @file profile_defs.h */

#define UCS_PROFILE_STACK_MAX 64


/**
 * Profiling modes
 */
enum {
    UCS_PROFILE_MODE_ACCUM, /**< Accumulate elapsed time per location */
    UCS_PROFILE_MODE_LOG,   /**< Record all events */
    UCS_PROFILE_MODE_LAST
};


/**
 * Profiling location type
 */
typedef enum {
    UCS_PROFILE_TYPE_SAMPLE,        /**< Sample only */
    UCS_PROFILE_TYPE_SCOPE_BEGIN,   /**< Begin a scope */
    UCS_PROFILE_TYPE_SCOPE_END,     /**< End a scope */
    UCS_PROFILE_TYPE_REQUEST_NEW,   /**< New asynchronous request */
    UCS_PROFILE_TYPE_REQUEST_EVENT, /**< Some progress is made on a request */
    UCS_PROFILE_TYPE_REQUEST_FREE,  /**< Asynchronous request released */
    UCS_PROFILE_TYPE_LAST
} ucs_profile_type_t;


/**
 * Profile output file header
 */
typedef struct ucs_profile_header {
    char                     cmdline[1024]; /**< Command line */
    char                     hostname[40];  /**< Host name */
    uint32_t                 pid;           /**< Process ID */
    uint32_t                 mode;          /**< Profiling mode */
    uint32_t                 num_locations; /**< Number of locations in the file */
    uint64_t                 num_records;   /**< Number of records in the file */
    uint64_t                 one_second;    /**< How much time is one second on the sampled machine */
} UCS_S_PACKED ucs_profile_header_t;


/**
 * Profile output file sample record
 */
typedef struct ucs_profile_record {
    uint64_t                 timestamp;     /**< Record timestamp */
    uint64_t                 param64;       /**< Custom 64-bit parameter */
    uint32_t                 param32;       /**< Custom 32-bit parameter */
    uint32_t                 location;      /**< Location identifier */
} UCS_S_PACKED ucs_profile_record_t;


/**
 * Profile location record
 */
typedef struct ucs_profile_location {
    char                     file[64];      /**< Source file name */
    char                     function[64];  /**< Function name */
    char                     name[32];      /**< User-provided name */
    volatile int             *loc_id_p;     /**< Back-pointer for location ID */
    int                      line;          /**< Source line number */
    uint8_t                  type;          /**< From ucs_profile_type_t */
    uint64_t                 total_time;    /**< Total interval from previous location */
    size_t                   count;         /**< Number of times we've hit this location */
} UCS_S_PACKED ucs_profile_location_t;


extern const char *ucs_profile_mode_names[];


/**
 * Initialize profiling system.
 */
void ucs_profile_global_init(void);


/**
 * Save and cleanup profiling.
 */
void ucs_profile_global_cleanup(void);


/**
 * Save and reset profiling.
 */
void ucs_profile_dump(void);

END_C_DECLS

#endif
