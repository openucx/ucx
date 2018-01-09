/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_PROFILE_H_
#define UCS_PROFILE_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/config/global_opts.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/time/time.h>


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
    int                      *loc_id_p;     /**< Back-pointer for location ID */
    int                      line;          /**< Source line number */
    uint8_t                  type;          /**< From ucs_profile_type_t */
    uint64_t                 total_time;    /**< Total interval from previous location */
    size_t                   count;         /**< Number of times we've hit this location */
} UCS_S_PACKED ucs_profile_location_t;


/**
 * Profiling global context
 */
typedef struct ucs_profile_global_context {

    ucs_profile_location_t   *locations;    /**< Array of all locations */
    unsigned                 num_locations; /**< Number of valid locations */
    unsigned                 max_locations; /**< Size of locations array */

    struct {
        ucs_profile_record_t *start, *end;  /**< Circular log buffer */
        ucs_profile_record_t *current;      /**< Current log pointer */
        int                  wraparound;    /**< Whether log was rotated */
    } log;

    struct {
        int                  stack_top;     /**< Index of stack top */
        ucs_time_t           stack[UCS_PROFILE_STACK_MAX]; /**< Timestamps for each nested scope */
    } accum;

} ucs_profile_global_context_t;


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


#if HAVE_PROFILING

extern const char *ucs_profile_mode_names[];

/*
 * Register a profiling location - should be called once per location in the
 * code, before the first record of each such location is made.
 * Should not be used directly - use UCS_PROFILE macros instead.
 *
 * @param [in]  type      Location type.
 * @param [in]  file      Source file name.
 * @param [in]  line      Source line number.
 * @param [in]  function  Calling function name.
 * @param [in]  name      Location name.
 * @param [out] loc_id_p  Filled with location ID:
 *                          0   - profiling is disabled
 *                          >0  - location index + 1
 */
void ucs_profile_get_location(ucs_profile_type_t type, const char *name,
                              const char *file, int line, const char *function,
                              int *loc_id_p);


/*
 * Store a new record with the given data.
 * Should not be used directly - use UCS_PROFILE macros instead.
 *
 * @param [in]     type        Location type.
 * @param [in]     name        Location name.
 * @param [in]     param32     custom 32-bit parameter.
 * @param [in]     param64     custom 64-bit parameter.
 * @param [in]     file        Source file name.
 * @param [in]     line        Source line number.
 * @param [in]     function    Calling function name.
 * @param [in,out] loc_id_p    Variable used to maintain the location ID.
 */
static inline void ucs_profile_record(ucs_profile_type_t type, const char *name,
                                      uint32_t param32, uint64_t param64,
                                      const char *file, int line,
                                      const char *function, int *loc_id_p)
{
    extern ucs_profile_global_context_t ucs_profile_ctx;
    ucs_profile_global_context_t *ctx = &ucs_profile_ctx;
    ucs_profile_record_t   *rec;
    ucs_profile_location_t *loc;
    ucs_time_t current_time;
    int loc_id;

retry:
    loc_id = *loc_id_p;
    if (ucs_likely(loc_id == 0)) {
        return;
    }

    if (ucs_unlikely(loc_id == -1)) {
        ucs_profile_get_location(type, name, file, line, function, loc_id_p);
        goto retry;
    }

    current_time = ucs_get_time();
    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        loc              = &ctx->locations[loc_id - 1];
        switch (type) {
        case UCS_PROFILE_TYPE_SCOPE_BEGIN:
            ctx->accum.stack[++ctx->accum.stack_top] = current_time;
            ucs_assert(ctx->accum.stack_top < UCS_PROFILE_STACK_MAX);
            break;
        case UCS_PROFILE_TYPE_SCOPE_END:
            ucs_assert(ctx->accum.stack_top >= 0);
            loc->total_time += current_time - ctx->accum.stack[ctx->accum.stack_top];
            --ctx->accum.stack_top;
            break;
        default:
            break;
        }
        ++loc->count;
    }
    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        rec              = ctx->log.current;
        rec->timestamp   = current_time;
        rec->param64     = param64;
        rec->param32     = param32;
        rec->location    = loc_id - 1;
        if (++ctx->log.current >= ctx->log.end) {
            ctx->log.current    = ctx->log.start;
            ctx->log.wraparound = 1;
        }
    }
}

/* Helper macro */
#define _UCS_PROFILE_RECORD(_type, _name, _param64, _param32, _loc_id_p) \
    ucs_profile_record((_type), (_name), (_param64), (_param32),  __FILE__, \
                       __LINE__, __FUNCTION__, (_loc_id_p))


/* Helper macro */
#define __UCS_PROFILE_CODE(_name, _loop_var) \
    int _loop_var ; \
    for (({ UCS_PROFILE_SCOPE_BEGIN(); _loop_var = 1;}); \
         _loop_var; \
         ({ UCS_PROFILE_SCOPE_END(_name); _loop_var = 0;}))


/* Helper macro */
#define _UCS_PROFILE_CODE(_name, _var_suffix) \
    __UCS_PROFILE_CODE(_name, UCS_PP_TOKENPASTE(loop, _var_suffix))


/**
 * Record a profiling event.
 *
 * @param _type     Event type.
 * @param _name     Event name.
 * @param _param32  Custom 32-bit parameter.
 * @param _param64  Custom 64-bit parameter.
 */
#define UCS_PROFILE(_type, _name, _param32, _param64) \
    { \
        static int loc_id = -1; \
        _UCS_PROFILE_RECORD((_type), (_name), (_param32), (_param64), &loc_id); \
    }


/**
 * Record a profiling sample event.
 *
 * @param _name   Event name.
 */
#define UCS_PROFILE_SAMPLE(_name) \
    UCS_PROFILE(UCS_PROFILE_TYPE_SAMPLE, (_name), 0, 0)


/**
 * Record a scope-begin profiling event.
 */
#define UCS_PROFILE_SCOPE_BEGIN() \
    { \
        UCS_PROFILE(UCS_PROFILE_TYPE_SCOPE_BEGIN, "", 0, 0); \
        ucs_compiler_fence(); \
    }


/**
 * Record a scope-end profiling event.
 *
 * @param _name   Scope name.
 */
#define UCS_PROFILE_SCOPE_END(_name) \
    { \
        ucs_compiler_fence(); \
        UCS_PROFILE(UCS_PROFILE_TYPE_SCOPE_END, _name, 0, 0); \
    }


/**
 * Declare a profiled scope of code.
 *
 * Usage:
 *  UCS_PROFILE_CODE(<name>) {
 *     <code>
 *  }
 *
 * @param _name   Scope name.
 */
#define UCS_PROFILE_CODE(_name) \
    _UCS_PROFILE_CODE(_name, UCS_PP_UNIQUE_ID)


/**
 * Create a profiled function.
 *
 * Usage:
 *  UCS_PROFILE_FUNC(<retval>, <name>, (a, b), int a, char b)
 *
 * @param _ret_type   Function return type.
 * @param _name       Function name.
 * @param _arglist    List of argument *names* only.
 * @param ...         Argument declarations (with types).
 */
#define UCS_PROFILE_FUNC(_ret_type, _name, _arglist, ...) \
    static UCS_F_ALWAYS_INLINE _ret_type _name##_inner(__VA_ARGS__); \
    \
    _ret_type _name(__VA_ARGS__) { \
        UCS_PROFILE_SCOPE_BEGIN(); \
        _ret_type _ret = _name##_inner _arglist; \
        UCS_PROFILE_SCOPE_END(#_name); \
        return _ret; \
    } \
    static UCS_F_ALWAYS_INLINE _ret_type _name##_inner(__VA_ARGS__)


/**
 * Create a profiled function whose return type is void.
 *
 * Usage:
 *  UCS_PROFILE_FUNC_VOID(<name>, (a, b), int a, char b)
 *
 * @param _name       Function name.
 * @param _arglist    List of argument *names* only.
 * @param ...         Argument declarations (with types).
 */
#define UCS_PROFILE_FUNC_VOID(_name, _arglist, ...) \
    static UCS_F_ALWAYS_INLINE void _name##_inner(__VA_ARGS__); \
    \
    void _name(__VA_ARGS__) { \
        UCS_PROFILE_SCOPE_BEGIN(); \
        _name##_inner _arglist; \
        UCS_PROFILE_SCOPE_END(#_name); \
    } \
    static UCS_F_ALWAYS_INLINE void _name##_inner(__VA_ARGS__)


/*
 * Profile a function call, and specify explicit name string for the profile.
 * Useful when calling a function by a pointer.
 *
 * Usage:
 *  UCS_PROFILE_NAMED_CALL("name", function, arg1, arg2)
 *
 * @param _name   Name string for the profile.
 * @param _func   Function name.
 * @param ...     Function call arguments.
 */
#define UCS_PROFILE_NAMED_CALL(_name, _func, ...) \
    ({ \
        typeof(_func(__VA_ARGS__)) retval; \
        UCS_PROFILE_SCOPE_BEGIN(); \
        retval = _func(__VA_ARGS__); \
        UCS_PROFILE_SCOPE_END(_name); \
        retval; \
    })


/*
 * Profile a function call.
 *
 * Usage:
 *  UCS_PROFILE_CALL(function, arg1, arg2)
 *
 * @param _func   Function name.
 * @param ...     Function call arguments.
 */
#define UCS_PROFILE_CALL(_func, ...) \
    UCS_PROFILE_NAMED_CALL(#_func, _func, ## __VA_ARGS__)


/*
 * Profile a function call which does not return a value, and specify explicit
 * name string for the profile. Useful when calling a function by a pointer.
 *
 * Usage:
 *  UCS_PROFILE_NAMED_CALL_VOID("name", function, arg1, arg2)
 *
 * @param _name   Name string for the profile.
 * @param _func   Function name.
 * @param ...     Function call arguments.
 */
#define UCS_PROFILE_NAMED_CALL_VOID(_name, _func, ...) \
    { \
        UCS_PROFILE_SCOPE_BEGIN(); \
        _func(__VA_ARGS__); \
        UCS_PROFILE_SCOPE_END(_name); \
    }


/*
 * Profile a function call which does not return a value.
 *
 * Usage:
 *  UCS_PROFILE_CALL_VOID(function, arg1, arg2)
 *
 * @param _func   Function name.
 * @param ...     Function call arguments.
 */
#define UCS_PROFILE_CALL_VOID(_func, ...) \
    UCS_PROFILE_NAMED_CALL_VOID(#_func, _func, ## __VA_ARGS__)


/*
 * Profile a new request allocation.
 *
 * @param _req      Request pointer.
 * @param _name     Allocation site name.
 * @param _param32  Custom 32-bit parameter.
 */
#define UCS_PROFILE_REQUEST_NEW(_req, _name, _param32) \
    UCS_PROFILE(UCS_PROFILE_TYPE_REQUEST_NEW, (_name), (_param32), (uintptr_t)(_req));


/*
 * Profile a request progress event.
 *
 * @param _req      Request pointer.
 * @param _name     Event name.
 * @param _param32  Custom 32-bit parameter.
 */
#define UCS_PROFILE_REQUEST_EVENT(_req, _name, _param32) \
    UCS_PROFILE(UCS_PROFILE_TYPE_REQUEST_EVENT, (_name), (_param32), (uintptr_t)(_req));


/*
 * Profile a request progress event with status check.
 *
 * @param _req      Request pointer.
 * @param _name     Event name.
 * @param _param32  Custom 32-bit parameter.
 * @param _status   Status of the last progress event.
 */
#define UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(_req, _name, _param32, _status) \
    if (!UCS_STATUS_IS_ERR(_status)) { \
        UCS_PROFILE_REQUEST_EVENT((_req), (_name), (_param32)); \
    }


/*
 * Profile a request release.
 *
 * @param _req      Request pointer.
 */
#define UCS_PROFILE_REQUEST_FREE(_req) \
    UCS_PROFILE(UCS_PROFILE_TYPE_REQUEST_FREE, "", 0, (uintptr_t)(_req));


#else

#define UCS_PROFILE(...)                                    UCS_EMPTY_STATEMENT
#define UCS_PROFILE_SAMPLE(_name)                           UCS_EMPTY_STATEMENT
#define UCS_PROFILE_SCOPE_BEGIN()                           UCS_EMPTY_STATEMENT
#define UCS_PROFILE_SCOPE_END(_name)                        UCS_EMPTY_STATEMENT
#define UCS_PROFILE_CODE(_name)
#define UCS_PROFILE_FUNC(_ret_type, _name, _arglist, ...)   _ret_type _name(__VA_ARGS__)
#define UCS_PROFILE_FUNC_VOID(_name, _arglist, ...)         void _name(__VA_ARGS__)
#define UCS_PROFILE_NAMED_CALL(_name, _func, ...)           _func(__VA_ARGS__)
#define UCS_PROFILE_CALL(_func, ...)                        _func(__VA_ARGS__)
#define UCS_PROFILE_NAMED_CALL_VOID(_name, _func, ...)      _func(__VA_ARGS__)
#define UCS_PROFILE_CALL_VOID(_func, ...)                   _func(__VA_ARGS__)
#define UCS_PROFILE_REQUEST_NEW(...)                        UCS_EMPTY_STATEMENT
#define UCS_PROFILE_REQUEST_EVENT(...)                      UCS_EMPTY_STATEMENT
#define UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(...)         UCS_EMPTY_STATEMENT
#define UCS_PROFILE_REQUEST_FREE(...)                       UCS_EMPTY_STATEMENT

#endif

#endif
