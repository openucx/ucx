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
#define UCM_DEFINE_REPLACE_FUNC(_name, _rettype, _fail_val, ...) \
    \
    _rettype ucm_override_##_name(UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)); \
    \
    /* Call the original function using dlsym(RTLD_NEXT) */ \
    _rettype ucm_orig_##_name(UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
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
                                               ucm_override_##_name); \
            ucm_reloc_get_orig_thread = -1; \
            pthread_mutex_unlock(&ucm_reloc_get_orig_lock); \
        } \
        return orig_func_ptr(UCM_FUNC_PASS_ARGS(__VA_ARGS__)); \
    } \
    \
    /* Define a symbol which goes to the replacement - in case we are loaded first */ \
    _rettype ucm_override_##_name(UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        ucm_trace("%s()", __FUNCTION__); \
        \
        if (ucs_unlikely(ucm_reloc_get_orig_thread == pthread_self())) { \
            return _fail_val; \
        } \
        return ucm_##_name(UCM_FUNC_PASS_ARGS(__VA_ARGS__)); \
    }

#define UCM_OVERRIDE_FUNC(_name, _rettype) \
    _rettype _name() __attribute__ ((alias ("ucm_override_" UCS_PP_QUOTE(_name)))); \


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
