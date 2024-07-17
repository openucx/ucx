/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_PROFILE_DEFS_H_
#define UCS_PROFILE_DEFS_H_

#include <ucs/config/global_opts.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/time/time_def.h>
#include <limits.h>
#include <unistd.h>

BEGIN_C_DECLS

#define UCS_PROFILE_STACK_MAX       64
#define UCS_PROFILE_FILE_VERSION    3u
#define UCS_PROFILE_LOC_ID_UNKNOWN  -1
#define UCS_PROFILE_LOC_ID_DISABLED 0

/* Minimum backwards compatible version */
#define UCS_PROFILE_FILE_MIN_VERSION 3u


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
 * < ucs_profile_location_t > * ucs_profile_header_t::num_locations
 * [
 *    < ucs_profile_thread_header_t >
 *    < ucs_profile_thread_location_t > * ucs_profile_header_t::num_locations
 *    < ucs_profile_record_t > * ucs_profile_thread_header_t::num_records
 *
 * ] * ucs_profile_thread_header_t::num_threads
 * <env variables string>
 */


/**
 * Profile output file block/section offset and size
*/
typedef struct {
    uint64_t offset;
    uint64_t size;
} UCS_S_PACKED ucs_profile_block_header_t;


/**
 * Profile output file header
 */
typedef struct ucs_profile_header {
    uint32_t                   version;        /**< File format version */
    uint32_t                   feature_flags;  /**< Feature flags bitmask */
    char                       ucs_path[1024]; /**< UCX library path*/
    char                       cmdline[1024];  /**< Command line */
    char                       hostname[64];   /**< Host name */
    uint32_t                   pid;            /**< Process ID */
    uint32_t                   mode;           /**< Bitmask of profiling modes */
    ucs_profile_block_header_t env_vars;       /**< Size and offset of environment variables string*/
    ucs_profile_block_header_t locations;      /**< Size and offset of locations*/
    ucs_profile_block_header_t threads;        /**< Size and offset of threads*/
    uint64_t one_second;                       /**< How much time is one second on the sampled machine */
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

typedef struct ucs_profile_context ucs_profile_context_t;
typedef short ucs_profile_loc_id_t;


extern const char *ucs_profile_mode_names[];
extern ucs_profile_context_t *ucs_profile_default_ctx;


/**
 * Initialize profiling system.
 *
 * @param [in]  profile_mode  Profiling mode.
 * @param [in]  file_name     Profiling file.
 * @param [in]  max_file_size Limit for profiling log size.
 * @param [out] ctx_p         Profile context.
 *
 * @return Status code.
 */
ucs_status_t ucs_profile_init(unsigned profile_mode, const char *file_name,
                              size_t max_file_size, ucs_profile_context_t **ctx_p);


/**
 * Save and cleanup profiling.
 *
 * @param [in] ctx       Profile context.
 */
void ucs_profile_cleanup(ucs_profile_context_t *ctx);


/**
 * Save and reset profiling.
 *
 * @param [in] ctx       Profile context.
 */
void ucs_profile_dump(ucs_profile_context_t *ctx);


/*
 * Store a new record with the given data.
 * SHOULD NOT be used directly - use UCS_PROFILE macros instead.
 *
 * @param [in]     ctx         Global profile context.
 * @param [in]     type        Location type.
 * @param [in]     name        Location name.
 * @param [in]     param32     custom 32-bit parameter.
 * @param [in]     param64     custom 64-bit parameter.
 * @param [in]     file        Source file name.
 * @param [in]     line        Source line number.
 * @param [in]     function    Calling function name.
 * @param [in,out] loc_id_p    Variable used to maintain the location ID.
 */
void ucs_profile_record(ucs_profile_context_t *ctx, ucs_profile_type_t type,
                        const char *name, uint32_t param32, uint64_t param64,
                        const char *file, int line, const char *function,
                        volatile ucs_profile_loc_id_t *loc_id_p);


/**
 * Calculates number of threads in the profiled execution 
 * given total number of profiling records, num of location and threads
 * 
 * @param   total_num_records   accumulated num of records from all threads
 * @param   header              profiling header struct 
 * 
 * @return  number of threads in the profiled code
 */
unsigned ucs_profile_calc_num_threads(size_t total_num_records,
                                      const ucs_profile_header_t *header);

/**
 * Record a profiling event.
 *
 * @param _ctx      Profiling context.
 * @param _type     Event type.
 * @param _name     Event name.
 * @param _param32  Custom 32-bit parameter.
 * @param _param64  Custom 64-bit parameter.
 */
#define UCS_PROFILE_CTX_RECORD_ALWAYS(_ctx, _type, _name, _param32, _param64) \
    { \
        static ucs_profile_loc_id_t loc_id = UCS_PROFILE_LOC_ID_UNKNOWN; \
        if (loc_id != UCS_PROFILE_LOC_ID_DISABLED) { \
            ucs_profile_record(_ctx, _type, _name, _param32, _param64, \
                               __FILE__, __LINE__, __func__, &loc_id); \
        } \
    }


/**
 * Profile a block of code.
 *
 * @param _ctx      Profiling context.
 * @param _name     Event name.
 * @param _code     Code block to run and profile.
 */
#define UCS_PROFILE_CTX_CODE_ALWAYS(_ctx, _name, _code) \
    { \
        UCS_PROFILE_CTX_RECORD_ALWAYS(_ctx, UCS_PROFILE_TYPE_SCOPE_BEGIN, "", \
                                      0, 0); \
        ucs_compiler_fence(); \
        _code; \
        ucs_compiler_fence(); \
        UCS_PROFILE_CTX_RECORD_ALWAYS(_ctx, UCS_PROFILE_TYPE_SCOPE_END, _name, \
                                      0, 0); \
    }


/**
 * Create a profiled function.
 *
 * Usage:
 *  UCS_PROFILE_CTX_FUNC_ALWAYS(ctx, <retval>, <name>, (a, b), int a, char b)
 *
 * @param _ctx        Profiling context.
 * @param _ret_type   Function return type.
 * @param _name       Function name.
 * @param _arglist    List of argument *names* only.
 * @param ...         Argument declarations (with types).
 */
#define UCS_PROFILE_CTX_FUNC_ALWAYS(_ctx, _ret_type, _name, _arglist, ...) \
    static UCS_F_ALWAYS_INLINE _ret_type _name##_inner(__VA_ARGS__); \
    \
    _ret_type _name(__VA_ARGS__) \
    { \
        _ret_type _ret; \
        \
        UCS_PROFILE_CTX_CODE_ALWAYS(_ctx, #_name, \
                                    _ret = _name##_inner _arglist); \
        return _ret; \
    } \
    static UCS_F_ALWAYS_INLINE _ret_type _name##_inner(__VA_ARGS__)


/**
 * Create a profiled function whose return type is void.
 *
 * Usage:
 *  UCS_PROFILE_CTX_FUNC_VOID_ALWAYS(ctx, <name>, (a, b), int a, char b)
 *
 * @param _ctx        Profiling context.
 * @param _name       Function name.
 * @param _arglist    List of argument *names* only.
 * @param ...         Argument declarations (with types).
 */
#define UCS_PROFILE_CTX_FUNC_VOID_ALWAYS(_ctx, _name, _arglist, ...) \
    static UCS_F_ALWAYS_INLINE void _name##_inner(__VA_ARGS__); \
    \
    void _name(__VA_ARGS__) \
    { \
        UCS_PROFILE_CTX_CODE_ALWAYS(_ctx, #_name, _name##_inner _arglist); \
    } \
    static UCS_F_ALWAYS_INLINE void _name##_inner(__VA_ARGS__)


/*
 * Profile a function call, and specify explicit name string for the profile.
 * Useful when calling a function by a pointer. Uses default profile context.
 *
 * Usage:
 *  UCS_PROFILE_CTX_NAMED_CALL(ctx, "name", function, arg1, arg2)
 *
 * @param _name   Name string for the profile.
 * @param _func   Function name.
 * @param ...     Function call arguments.
 */
#define UCS_PROFILE_CTX_NAMED_CALL_ALWAYS(_ctx, _name, _func, ...) \
    ({ \
        ucs_typeof(_func(__VA_ARGS__)) retval; \
        \
        UCS_PROFILE_CTX_CODE_ALWAYS(_ctx, _name, retval = _func(__VA_ARGS__)); \
        retval; \
    })


/**
 * Record a profiling sample event.
 *
 * @param _name   Event name.
 */
#define UCS_PROFILE_SAMPLE_ALWAYS(_name) \
    UCS_PROFILE_CTX_RECORD_ALWAYS(ucs_profile_default_ctx, \
                                  UCS_PROFILE_TYPE_SAMPLE, (_name), 0, 0)


/**
 * Declare a profiled scope of code.
 *
 * Usage:
 *  UCS_PROFILE_CODE_ALWAYS(<name>, <code>)
 *
 * @param _name   Scope name.
 */
#define UCS_PROFILE_CODE_ALWAYS(_name, _code) \
    UCS_PROFILE_CTX_CODE_ALWAYS(ucs_profile_default_ctx, _name, _code)


/**
 * Create a profiled function. Uses default profile context.
 *
 * Usage:
 *  UCS_PROFILE_FUNC_ALWAYS(<retval>, <name>, (a, b), int a, char b)
 *
 * @param _ret_type   Function return type.
 * @param _name       Function name.
 * @param _arglist    List of argument *names* only.
 * @param ...         Argument declarations (with types).
 */
#define UCS_PROFILE_FUNC_ALWAYS(_ret_type, _name, _arglist, ...) \
    UCS_PROFILE_CTX_FUNC_ALWAYS(ucs_profile_default_ctx, _ret_type, _name, \
                                _arglist, ##__VA_ARGS__)


/**
 * Create a profiled function whose return type is void. Uses default profile
 * context.
 *
 * Usage:
 *  UCS_PROFILE_FUNC_VOID_ALWAYS(<name>, (a, b), int a, char b)
 *
 * @param _name       Function name.
 * @param _arglist    List of argument *names* only.
 * @param ...         Argument declarations (with types).
 */
#define UCS_PROFILE_FUNC_VOID_ALWAYS(_name, _arglist, ...) \
    UCS_PROFILE_CTX_FUNC_VOID_ALWAYS(ucs_profile_default_ctx, _name, _arglist, \
                                     ##__VA_ARGS__)


/*
 * Profile a function call, and specify explicit name string for the event.
 * Useful when calling a function by a pointer. Uses default profile context.
 *
 * Usage:
 *  ret = UCS_PROFILE_NAMED_CALL_ALWAYS("name", function, arg1, arg2)
 *
 * @param _name   Name string for the profile.
 * @param _func   Function name.
 * @param ...     Function call arguments.
 */
#define UCS_PROFILE_NAMED_CALL_ALWAYS(_name, _func, ...) \
    UCS_PROFILE_CTX_NAMED_CALL_ALWAYS(ucs_profile_default_ctx, _name, _func, \
                                      ##__VA_ARGS__)


/*
 * Profile a function call.
 *
 * Usage:
 *  ret = UCS_PROFILE_CALL_ALWAYS(function, arg1, arg2)
 *
 * @param _func   Function name.
 * @param ...     Function call arguments.
 */
#define UCS_PROFILE_CALL_ALWAYS(_func, ...) \
    UCS_PROFILE_NAMED_CALL_ALWAYS(#_func, _func, ##__VA_ARGS__)


/*
 * Profile a void function call, and specify explicit name string for the event.
 * Useful when calling a function by a pointer. Uses default profile context.
 *
 * Usage:
 *  UCS_PROFILE_NAMED_CALL_VOID_ALWAYS("name", function, arg1, arg2)
 *
 * @param _name   Name string for the profile.
 * @param _func   Function name.
 * @param ...     Function call arguments.
 */
#define UCS_PROFILE_NAMED_CALL_VOID_ALWAYS(_name, _func, ...) \
    UCS_PROFILE_CODE_ALWAYS(_name, _func(__VA_ARGS__))


/*
 * Profile a void function call.
 *
 * Usage:
 *  UCS_PROFILE_CALL_VOID_ALWAYS(function, arg1, arg2)
 *
 * @param _func   Function name.
 * @param ...     Function call arguments.
 */
#define UCS_PROFILE_CALL_VOID_ALWAYS(_func, ...) \
    UCS_PROFILE_NAMED_CALL_VOID_ALWAYS(#_func, _func, __VA_ARGS__)


/*
 * Profile a request progress event.
 *
 * @param _req      Request pointer.
 * @param _name     Event name.
 * @param _param32  Custom 32-bit parameter.
 */
#define UCS_PROFILE_REQUEST_EVENT_ALWAYS(_req, _name, _param32) \
    UCS_PROFILE_CTX_RECORD_ALWAYS(ucs_profile_default_ctx, \
                                  UCS_PROFILE_TYPE_REQUEST_EVENT, (_name), \
                                  (_param32), (uintptr_t)(_req));

END_C_DECLS

#endif
