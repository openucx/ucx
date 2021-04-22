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
#include <limits.h>
#include <stdint.h>

BEGIN_C_DECLS

/** @file profile_defs.h */

#define UCS_PROFILE_STACK_MAX     64
#define UCS_PROFILE_FILE_VERSION  2u


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


/*
 * Profile file structure:
 *
 * < ucs_profile_header_t >
 * < ucs_profile_location_t > * ucs_profile_header_t::num_locaitons
 * [
 *    < ucs_profile_thread_header_t >
 *    < ucs_profile_thread_location_t > * ucs_profile_header_t::num_locaitons
 *    < ucs_profile_record_t > * ucs_profile_thread_header_t::num_records
 *
 * ] * ucs_profile_thread_header_t::num_threads
 */


/**
 * Profile output file header
 */
typedef struct ucs_profile_header {
    uint32_t                 version;       /**< File format version */
    char                     ucs_path[1024];/**< UCX library path*/
    char                     cmdline[1024]; /**< Command line */
    char                     hostname[64];  /**< Host name */
    uint32_t                 pid;           /**< Process ID */
    uint32_t                 mode;          /**< Bitmask of profiling modes */
    uint32_t                 num_locations; /**< Number of locations in the file */
    uint32_t                 num_threads;   /**< Number of threads in the file */
    uint64_t                 one_second;    /**< How much time is one second on the sampled machine */
} UCS_S_PACKED ucs_profile_header_t;


/**
 * Profile location record
 */
typedef struct ucs_profile_location {
    char                     file[64];      /**< Source file name */
    char                     function[64];  /**< Function name */
    char                     name[32];      /**< User-provided name */
    int                      line;          /**< Source line number */
    uint8_t                  type;          /**< From ucs_profile_type_t */
} UCS_S_PACKED ucs_profile_location_t;


/**
 * Profile output file thread header
 */
typedef struct ucs_profile_thread_header {
    uint32_t                 tid;            /**< System thread id */
    uint64_t                 start_time;     /**< Time of thread start */
    uint64_t                 end_time;       /**< Time of thread exit */
    uint64_t                 num_records;    /**< Number of records for the thread */
} UCS_S_PACKED ucs_profile_thread_header_t;


/**
 * Profile thread location with samples
 */
typedef struct ucs_profile_thread_location {
    uint64_t                 total_time;    /**< Total interval from previous location */
    size_t                   count;         /**< Number of times we've hit this location */
} UCS_S_PACKED ucs_profile_thread_location_t;


/**
 * Profile output file sample record
 */
typedef struct ucs_profile_record {
    uint64_t                 timestamp;     /**< Record timestamp */
    uint64_t                 param64;       /**< Custom 64-bit parameter */
    uint32_t                 param32;       /**< Custom 32-bit parameter */
    uint32_t                 location;      /**< Location identifier */
} UCS_S_PACKED ucs_profile_record_t;


extern const char *ucs_profile_mode_names[];


typedef uint64_t (*ucs_profile_range_start_t)(const char *format, ...);

typedef void (*ucs_profile_range_stop_t)(uint64_t id);

typedef void (*ucs_profile_range_push_t)(const char *format, ...);

typedef void (*ucs_profile_range_pop_t)();

typedef void (*ucs_profile_range_add_marker_t)(const char *format, ...);

/**
 * Profile range operations
 */
typedef struct ucs_profile_range_ops {
    ucs_profile_range_start_t      start;
    ucs_profile_range_stop_t       stop;
    ucs_profile_range_push_t       push;
    ucs_profile_range_pop_t        pop;
    ucs_profile_range_add_marker_t add_marker;
} ucs_profile_range_ops_t;

extern ucs_profile_range_ops_t ucs_profile_range_fxns;

/**
 * Initialize profiling system.
 */
void ucs_profile_global_init();


/**
 * Save and cleanup profiling.
 */
void ucs_profile_global_cleanup();


/**
 * Save and reset profiling.
 */
void ucs_profile_dump();


/*
 * Start a range trace on an arbitrary event in a potentially nested fashion.
 * A range tracing can be started in a function and potentially end in an
 * another function or a recursive invocation of the same function.
 * A unique ID is returned to stop tracing the event.
 *
 * @param [in]     format      String name for the range.
 *
 * @return ID to be used to stop tracing a range
 */
uint64_t ucs_profile_range_start(const char *format, ...);


/*
 * Stop range trace.
 *
 * @param [in]     id          id that was returned from range start.
 *
 */
void ucs_profile_range_stop(uint64_t id);


/*
 * Add a marker on trace profiles.
 *
 * @param [in]     format      String name for the marker.
 *
 */
void ucs_profile_range_add_marker(const char *format, ...);


/*
 * Start a range trace in a non-nested fashion. Range tracing must start and end
 * in the same function.
 *
 * @param [in]     format      String name for the marker.
 *
 */
void ucs_profile_range_push(const char *format, ...);


/*
 * Stop a range trace in a non-nested fashion.
 *
 */
void ucs_profile_range_pop();

END_C_DECLS

#endif
