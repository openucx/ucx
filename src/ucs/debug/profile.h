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

#include <ucs/sys/preprocessor.h>
#include <ucs/time/time.h>
#include <ucs/debug/log.h>


#define UCS_PROFILE_STACK_MAX 64


/**
 * Profiling modes
 */
enum {
    UCS_PROFILE_MODE_ACCUM, /* Accumulate elapsed time per location */
    UCS_PROFILE_MODE_LOG,   /* Record all events */
    UCS_PROFILE_MODE_LAST
};


/**
 * Profiling location type
 */
typedef enum {
    UCS_PROFILE_TYPE_SAMPLE,        /* Sample only */
    UCS_PROFILE_TYPE_SCOPE_BEGIN,   /* Begin a scope */
    UCS_PROFILE_TYPE_SCOPE_END,     /* End a scope */
    UCS_PROFILE_TYPE_LAST
} ucs_profile_type_t;


/**
 * Profile output file header
 */
typedef struct ucs_profile_header {
    char                     cmdline[1024]; /* Command line */
    char                     hostname[40];  /* Host name */
    uint32_t                 pid;           /* Process ID */
    uint32_t                 mode;          /* Profiling mode */
    uint32_t                 num_locations; /* Number of locations in the file */
    uint64_t                 num_records;   /* Number of records in the file */
    uint64_t                 one_second;    /* How much time is one second on the sampled machine */
} UCS_S_PACKED ucs_profile_header_t;


/**
 * Profile output file sample record
 */
typedef struct ucs_profile_record {
    uint64_t                 timestamp;     /* Record timestamp */
    uint32_t                 location;      /* Location identifier */
} UCS_S_PACKED ucs_profile_record_t;


/**
 * Profile location record
 */
typedef struct ucs_profile_location {
    char                     file[64];      /* Source file name */
    char                     function[64];  /* Function name */
    char                     name[32];      /* User-provided name */
    int                      line;          /* Source line number */
    uint8_t                  type;          /* From ucs_profile_type_t */
    uint64_t                 total_time;    /* Total interval from previous location */
    size_t                   count;         /* Number of times we've hit this location */
} UCS_S_PACKED ucs_profile_location_t;


/**
 * Profiling global context
 */
typedef struct ucs_profile_global_context {

    ucs_profile_location_t   *locations;    /* Array of all locations */
    unsigned                 num_locations; /* Number of valid locations */
    unsigned                 max_locations; /* Size of locations array */

    struct {
        ucs_profile_record_t *start, *end;  /* Circular log buffer */
        ucs_profile_record_t *current;      /* Current log pointer */
        int                  wraparound;    /* Whether log was rotated */
    } log;

    struct {
        int                  stack_top;     /* Index of stack top */
        ucs_time_t           stack[UCS_PROFILE_STACK_MAX]; /* Timestamps for each nested scope */
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
 *
 * @return 0 for disabled record, positive instrumentation record id otherwise.
 */
int ucs_profile_get_location(ucs_profile_type_t type, const char *file, int line,
                             const char *function, const char *name);


/*
 * Store a new record with the given data.
 */
static inline void ucs_profile_record(ucs_profile_type_t type, const char *name,
                                      int *location_p, const char *file,
                                      int line, const char *function)
{
    extern ucs_profile_global_context_t ucs_profile_ctx;
    ucs_profile_global_context_t *ctx = &ucs_profile_ctx;
    ucs_profile_record_t   *rec;
    ucs_profile_location_t *loc;
    ucs_time_t current_time;

retry:
    if (ucs_likely(*location_p == 0)) {
        return;
    }

    if (ucs_unlikely(*location_p == -1)) {
        *location_p = ucs_profile_get_location(type, file, line, function, name);
        goto retry;
    }

    current_time = ucs_get_time();
    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        loc              = &ctx->locations[*location_p - 1];
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
        rec->location    = *location_p - 1;
        if (++ctx->log.current >= ctx->log.end) {
            ctx->log.current    = ctx->log.start;
            ctx->log.wraparound = 1;
        }
    }
}

/* Helper marco */
#define _UCS_PROFILE_RECORD(_type, _name, _location_p) \
    ucs_profile_record((_type), (_name), (_location_p), \
                       __FILE__, __LINE__, __FUNCTION__) \


/* Helper marco */
#define __UCS_PROFILE_CODE(_name, _loop) \
    int _loop_var ; \
    for (({ UCS_PROFILE_SCOPE_BEGIN(); _loop_var = 1;}); \
         _loop_var; \
         ({ UCS_PROFILE_SCOPE_END(_name); _loop_var = 0;}))


/* Helper marco */
#define _UCS_PROFILE_CODE(_name, _var_suffix) \
    __UCS_PROFILE_CODE(_name, UCS_PP_TOKENPASTE(loop, _var_suffix))


/**
 * Record a profiling event.
 *
 * @param _type   Event type.
 * @param _name   Event name.
 */
#define UCS_PROFILE(_type, _name) \
    { \
        static int location = -1; \
        _UCS_PROFILE_RECORD((_type), (_name), &location); \
    }


#define UCS_PROFILE_SAMPLE(_name) \
    UCS_PROFILE(UCS_PROFILE_TYPE_SAMPLE, (_name))


/**
 * Record a scope-begin profiling event.
 */
#define UCS_PROFILE_SCOPE_BEGIN() \
    { \
        UCS_PROFILE(UCS_PROFILE_TYPE_SCOPE_BEGIN, ""); \
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
        UCS_PROFILE(UCS_PROFILE_TYPE_SCOPE_END, _name); \
    }


/**
 * Declare a profiled scope of code.
 *
 * Usage:
 *  UCS_PROFILE_CODE(<name>) {
 *     <code>
 *  }
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
 *  UCS_PROFILE_FUNC(<retval>, <name>, (a, b), int a, char b)
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
 * Profile a function call.
 *
 * @param _func   Function name.
 * @param ...     Function call arguments.
 */
#define UCS_PROFILE_CALL(_func, ...) \
    ({ \
        typeof(_func(__VA_ARGS__)) retval; \
        UCS_PROFILE_SCOPE_BEGIN(); \
        retval = _func(__VA_ARGS__); \
        UCS_PROFILE_SCOPE_END(#_func); \
        retval; \
    })


/*
 * Profile a function call which does not return a value..
 *
 * @param _func   Function name.
 * @param ...     Function call arguments.
 */
#define UCS_PROFILE_CALL_VOID(_func, ...) \
    { \
        UCS_PROFILE_SCOPE_BEGIN(); \
        _func(__VA_ARGS__); \
        UCS_PROFILE_SCOPE_END(#_func); \
    }


#else

#define UCS_PROFILE(...)                                    UCS_EMPTY_STATEMENT
#define UCS_PROFILE_SAMPLE(_name)                           UCS_EMPTY_STATEMENT
#define UCS_PROFILE_SCOPE_BEGIN()                           UCS_EMPTY_STATEMENT
#define UCS_PROFILE_SCOPE_END(_name)                        UCS_EMPTY_STATEMENT
#define UCS_PROFILE_CODE(_name)
#define UCS_PROFILE_FUNC(_ret_type, _name, _arglist, ...)   _ret_type _name(__VA_ARGS__)
#define UCS_PROFILE_CALL(_func, ...)                        _func(__VA_ARGS__)
#define UCS_PROFILE_CALL_VOID(_func, ...)                   _func(__VA_ARGS__)

#endif

#endif
