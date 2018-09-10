/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCM_UTIL_REPLACE_H_
#define UCM_UTIL_REPLACE_H_

#include <ucs/datastruct/list.h>
#include <ucs/type/status.h>
#include <pthread.h>

extern pthread_mutex_t ucm_reloc_get_orig_lock;
extern pthread_t volatile ucm_reloc_get_orig_thread;

/**
 * Define a replacement function to a memory-mapping function call, which calls
 * the event handler, and if event handler returns error code - calls the original
 * function.
 */

/* Due to CUDA API redifinition we have to create proxy macro to eliminate
 * redifinition of internal finction names */
#define UCM_DEFINE_REPLACE_FUNC(_name, _rettype, _fail_val, ...) \
    _UCM_DEFINE_REPLACE_FUNC(ucm_override_##_name, ucm_##_name, _rettype, _fail_val, __VA_ARGS__)

#define _UCM_DEFINE_REPLACE_FUNC(_over_name, _ucm_name, _rettype, _fail_val, ...) \
    \
    /* Define a symbol which goes to the replacement - in case we are loaded first */ \
    _rettype _over_name(UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        ucm_trace("%s()", __FUNCTION__); \
        \
        if (ucs_unlikely(ucm_reloc_get_orig_thread == pthread_self())) { \
            return _fail_val; \
        } \
        return _ucm_name(UCM_FUNC_PASS_ARGS(__VA_ARGS__)); \
    }

#define UCM_OVERRIDE_FUNC(_name, _rettype) \
    _rettype _name() __attribute__ ((alias (UCS_PP_QUOTE(ucm_override_##_name)))); \

#define UCM_DEFINE_DLSYM_FUNC(_name, _rettype, _fail_val, ...) \
    _UCM_DEFINE_DLSYM_FUNC(_name, ucm_orig_##_name, ucm_override_##_name, \
                          _rettype, _fail_val, __VA_ARGS__)

#define _UCM_DEFINE_DLSYM_FUNC(_name, _orig_name, _over_name, _rettype, _fail_val, ...) \
    _rettype _over_name(UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)); \
    \
    /* Call the original function using dlsym(RTLD_NEXT) */ \
    _rettype _orig_name(UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        typedef _rettype (*func_ptr_t) (__VA_ARGS__); \
        static func_ptr_t orig_func_ptr = NULL; \
        \
        ucm_trace("%s()", __FUNCTION__); \
        \
        if (ucs_unlikely(orig_func_ptr == NULL)) { \
            pthread_mutex_lock(&ucm_reloc_get_orig_lock); \
            ucm_reloc_get_orig_thread = pthread_self(); \
            orig_func_ptr = ucm_reloc_get_orig(UCS_PP_QUOTE(_name), \
                                               _over_name); \
            ucm_reloc_get_orig_thread = -1; \
            pthread_mutex_unlock(&ucm_reloc_get_orig_lock); \
        } \
        return orig_func_ptr(UCM_FUNC_PASS_ARGS(__VA_ARGS__)); \
    }

#define UCM_DEFINE_REPLACE_DLSYM_FUNC(_name, _rettype, _fail_val, ...) \
    _UCM_DEFINE_DLSYM_FUNC(_name, ucm_orig_##_name, ucm_override_##_name, \
                          _rettype, _fail_val, __VA_ARGS__) \
    _UCM_DEFINE_REPLACE_FUNC(ucm_override_##_name, ucm_##_name, \
                             _rettype, _fail_val, __VA_ARGS__)

#define UCM_DEFINE_SYSCALL_FUNC(_name, _rettype, _syscall_id, ...) \
    /* Call syscall */ \
    _rettype ucm_orig_##_name(UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        return (_rettype)syscall(_syscall_id, UCM_FUNC_PASS_ARGS(__VA_ARGS__)); \
    }

#define UCM_DEFINE_SELECT_FUNC(_name, _rettype, _fail_val, _syscall_id, ...) \
    _UCM_DEFINE_DLSYM_FUNC(_name, ucm_orig_##_name##_dlsym, ucm_override_##_name, \
                          _rettype, _fail_val, __VA_ARGS__) \
    UCM_DEFINE_SYSCALL_FUNC(_name##_syscall, _rettype, _syscall_id, __VA_ARGS__) \
    _rettype ucm_orig_##_name(UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        return ucm_global_opts.enable_syscall ? \
               ucm_orig_##_name##_syscall(UCM_FUNC_PASS_ARGS(__VA_ARGS__)) : \
               ucm_orig_##_name##_dlsym(UCM_FUNC_PASS_ARGS(__VA_ARGS__)); \
    }

/*
 * Define argument list with given types.
 */
#define UCM_FUNC_DEFINE_ARGS(...) \
    UCS_PP_FOREACH_SEP(_UCM_FUNC_ARG_DEFINE, _, \
                       UCS_PP_ZIP((UCS_PP_SEQ(UCS_PP_NUM_ARGS(__VA_ARGS__))), \
                                  (__VA_ARGS__)))

/*
 * Pass auto-generated arguments to a function call.
 */
#define UCM_FUNC_PASS_ARGS(...) \
    UCS_PP_FOREACH_SEP(_UCM_FUNC_ARG_PASS, _, UCS_PP_SEQ(UCS_PP_NUM_ARGS(__VA_ARGS__)))


/*
 * Helpers
 */
#define _UCM_FUNC_ARG_DEFINE(_, _bundle) \
    __UCM_FUNC_ARG_DEFINE(_, UCS_PP_TUPLE_0 _bundle, UCS_PP_TUPLE_1 _bundle)
#define __UCM_FUNC_ARG_DEFINE(_, _index, _type) \
    _type UCS_PP_TOKENPASTE(arg, _index)
#define _UCM_FUNC_ARG_PASS(_, _index) \
    UCS_PP_TOKENPASTE(arg, _index)

#endif
