/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_PROFILE_ON_H_
#define UCS_PROFILE_ON_H_

#include "profile_defs.h"

#include <ucs/sys/compiler_def.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/config/global_opts.h>


BEGIN_C_DECLS

/** @file profile_on.h */

/* Helper macro */
#define _UCS_PROFILE_RECORD(_type, _name, _param64, _param32, _loc_id_p) \
    { \
        if (*(_loc_id_p) != 0) { \
            ucs_profile_record((_type), (_name), (_param64), (_param32),  \
                               __FILE__, __LINE__, __FUNCTION__, (_loc_id_p)); \
        } \
    }


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
    _ret_type _name(__VA_ARGS__) \
    { \
        _ret_type _ret; \
        UCS_PROFILE_SCOPE_BEGIN(); \
        _ret = _name##_inner _arglist; \
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


/*
 * Store a new record with the given data.
 * SHOULD NOT be used directly - use UCS_PROFILE macros instead.
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
void ucs_profile_record(ucs_profile_type_t type, const char *name,
                        uint32_t param32, uint64_t param64, const char *file,
                        int line, const char *function, volatile int *loc_id_p);


/**
 * Reset the internal array of profiling locations.
 * Used for testing purposes only.
 */
void ucs_profile_reset_locations();


END_C_DECLS

#endif
