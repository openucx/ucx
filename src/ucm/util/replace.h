/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCM_UTIL_REPLACE_H_
#define UCM_UTIL_REPLACE_H_

#include <ucm/bistro/bistro.h>
#include <ucs/datastruct/list.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/type/status.h>
#include <pthread.h>

extern pthread_mutex_t ucm_reloc_get_orig_lock;
extern pthread_t volatile ucm_reloc_get_orig_thread;

/**
 * Define a replacement function to a memory-mapping function call, which calls
 * the event handler, and if event handler returns error code - calls the original
 * function.
 */

/* Due to CUDA API redefinition we have to create proxy macro to eliminate
 * redefinition of internal function names */
#define UCM_DEFINE_REPLACE_FUNC(_name, _rettype, _fail_val, ...) \
    _UCM_DEFINE_REPLACE_FUNC(ucm_override_##_name, ucm_##_name, _rettype, _fail_val, __VA_ARGS__)

#define _UCM_DEFINE_REPLACE_FUNC(_over_name, _ucm_name, _rettype, _fail_val, ...) \
    \
    /* Define a symbol which goes to the replacement - in case we are loaded first */ \
    _rettype _over_name(UCS_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        _rettype res; \
        UCM_BISTRO_PROLOGUE; \
        ucm_trace("%s()", __func__); \
        \
        if (ucs_unlikely(ucm_reloc_get_orig_thread == pthread_self())) { \
            return (_rettype)_fail_val; \
        } \
        res = _ucm_name(UCS_FUNC_PASS_ARGS(__VA_ARGS__)); \
        UCM_BISTRO_EPILOGUE; \
        return res; \
    }

#define UCM_DEFINE_DLSYM_FUNC(_name, _rettype, ...) \
    _UCM_DEFINE_DLSYM_FUNC(_name, ucm_orig_##_name, ucm_override_##_name, \
                           _rettype, __VA_ARGS__)


#define _UCM_DEFINE_DLSYM_FUNC(_name, _orig_name, _over_name, _rettype, ...) \
    _rettype _over_name(UCS_FUNC_DEFINE_ARGS(__VA_ARGS__)); \
    \
    /* Call the original function using dlsym(RTLD_NEXT) */ \
    _rettype _orig_name(UCS_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        typedef _rettype (*func_ptr_t) (__VA_ARGS__); \
        static func_ptr_t orig_func_ptr = NULL; \
        \
        ucm_trace("%s()", __func__); \
        \
        if (ucs_unlikely(orig_func_ptr == NULL)) { \
            pthread_mutex_lock(&ucm_reloc_get_orig_lock); \
            ucm_reloc_get_orig_thread = pthread_self(); \
            orig_func_ptr = (func_ptr_t)ucm_reloc_get_orig(UCS_PP_QUOTE(_name), \
                                                           _over_name); \
            ucm_reloc_get_orig_thread = (pthread_t)-1; \
            pthread_mutex_unlock(&ucm_reloc_get_orig_lock); \
        } \
        return orig_func_ptr(UCS_FUNC_PASS_ARGS(__VA_ARGS__)); \
    }

#define UCM_DEFINE_REPLACE_DLSYM_FUNC(_name, _rettype, _fail_val, ...) \
    _UCM_DEFINE_DLSYM_FUNC(_name, ucm_orig_##_name, ucm_override_##_name, \
                           _rettype, __VA_ARGS__) \
    _UCM_DEFINE_REPLACE_FUNC(ucm_override_##_name, ucm_##_name, \
                             _rettype, _fail_val, __VA_ARGS__)

/**
 * Defines the following:
 *  - ucm_orig_##_name##_dlsym - calls original function by symbol lookup
 *  - ucm_orig_##_name         - function pointer, initialized by default to
 *                               ucm_orig_##_name##_dlsym
 *  - ucm_override_##_name     - calls ucm_##_name
 */
#define UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(_name, _rettype, _fail_val, ...) \
    _UCM_DEFINE_DLSYM_FUNC(_name, ucm_orig_##_name##_dlsym, \
                           ucm_override_##_name, _rettype, __VA_ARGS__) \
    \
    _rettype (*ucm_orig_##_name)(UCS_FUNC_DEFINE_ARGS(__VA_ARGS__)) = \
        ucm_orig_##_name##_dlsym; \
    \
    _UCM_DEFINE_REPLACE_FUNC(ucm_override_##_name, ucm_##_name, \
                             _rettype, _fail_val, __VA_ARGS__)

#define UCM_DEFINE_SYSCALL_FUNC(_name, _rettype, _syscall_id, ...) \
    /* Call syscall */ \
    _rettype ucm_orig_##_name(UCS_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        return (_rettype)syscall(_syscall_id, UCS_FUNC_PASS_ARGS(__VA_ARGS__)); \
    }

#if UCM_BISTRO_HOOKS
#  define UCM_DEFINE_SELECT_FUNC(_name, _rettype, _syscall_id, ...) \
    _UCM_DEFINE_DLSYM_FUNC(_name, ucm_orig_##_name##_dlsym, \
                           ucm_override_##_name, _rettype, __VA_ARGS__) \
    UCM_DEFINE_SYSCALL_FUNC(_name##_syscall, _rettype, _syscall_id, __VA_ARGS__) \
    _rettype ucm_orig_##_name(UCS_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        return (ucm_mmap_hook_mode() == UCM_MMAP_HOOK_BISTRO) ? \
               ucm_orig_##_name##_syscall(UCS_FUNC_PASS_ARGS(__VA_ARGS__)) : \
               ucm_orig_##_name##_dlsym(UCS_FUNC_PASS_ARGS(__VA_ARGS__)); \
    }
#else
#  define UCM_DEFINE_SELECT_FUNC(_name, _rettype, _syscall_id, ...) \
    UCM_DEFINE_DLSYM_FUNC(_name, _rettype, __VA_ARGS__)
#endif

#endif
