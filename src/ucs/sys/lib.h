/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021-2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef LIB_H
#define LIB_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/compiler_def.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/config/types.h>
#include <ucs/type/status.h>

#include <stdint.h>
#include <dlfcn.h>


BEGIN_C_DECLS

/**
 * @return Full info on current library.
 */
ucs_status_t ucs_sys_get_lib_info(Dl_info *dl_info);


/**
 * @return Full path to current library.
 */
const char *ucs_sys_get_lib_path();


/**
 * @return UCS library loading address.
 */
unsigned long ucs_sys_get_lib_base_addr();


/**
 * Look up a symbol in an opened library, without searching its dependencies.
 *
 * @param [in] lib_path     Path of the library, for logging.
 * @param [in] dl           Handle of the opened library.
 * @param [in] symbol       Name of the symbol.
 * @param [in] log_level    Level of the messages about lookup failures.
 *
 * @return Symbol address, or NULL if the library does not define it.
 */
void *ucs_sys_dlsym_shallow(const char *lib_path, void *dl,
                            const char *symbol, ucs_log_level_t log_level);


/**
 * Look up an optional symbol in the already loaded library which contains
 * @a lib_addr. The symbol is returned only if that library defines it, and
 * the caller must keep the library loaded while using it.
 *
 * @param [in] lib_addr     Address of any symbol of the library to search.
 * @param [in] sym_name     Name of the optional symbol.
 *
 * @return Symbol address, or NULL if the library does not define it.
 */
void *ucs_sys_get_lib_sym(const void *lib_addr, const char *sym_name);


/**
 * Get an optional function defined by the library which contains
 * @a _lib_addr. Shared libraries look it up at runtime, so loading them does
 * not require the library to define it; objects built without PIC (static
 * archives) reference it directly.
 *
 * @param [in] _lib_addr  Function of the same library which the caller links,
 *                        so that the library stays loaded.
 * @param [in] _func      Optional function declared by the included headers.
 *
 * @return Pointer to @a _func, or NULL if the loaded library does not define
 *         it.
 */
#ifdef UCX_SHARED_LIB
#  define UCS_SYS_LIB_OPTIONAL_FUNC(_lib_addr, _func) \
      ((ucs_typeof(_func)*)ucs_sys_get_lib_sym((const void*)(_lib_addr), \
                                               #_func))
#else
#  define UCS_SYS_LIB_OPTIONAL_FUNC(_lib_addr, _func) (&(_func))
#endif


/**
 * Look up a symbol in an opened library, and log a debug message if it is not
 * found.
 *
 * @param [in] dl           Handle of the opened library.
 * @param [in] sym_name     Name of the symbol.
 *
 * @return Symbol address, or NULL if it is not found.
 */
void *ucs_sys_dlsym(void *dl, const char *sym_name);


/* Pointer to function @a _func, in a table of pointers named by @a _prefix */
#define UCS_SYS_LIB_FUNC_PTR(_prefix, _func) UCS_PP_TOKENPASTE(_prefix, _func)

#define UCS_SYS_LIB_FUNC_PTR_DECLARE(_prefix, _func) \
    extern ucs_typeof(_func) *UCS_SYS_LIB_FUNC_PTR(_prefix, _func)

#define UCS_SYS_LIB_FUNC_PTR_DEFINE(_prefix, _func) \
    ucs_typeof(_func) *UCS_SYS_LIB_FUNC_PTR(_prefix, _func) = NULL

/* Load the pointer from the library opened as @a _dl */
#define UCS_SYS_LIB_FUNC_LOAD(_prefix, _func, _dl) \
    UCS_SYS_LIB_FUNC_PTR(_prefix, _func) = \
            (ucs_typeof(_func)*)ucs_sys_dlsym(_dl, UCS_PP_MAKE_STRING(_func))

/* Load the pointer with @ref UCS_SYS_LIB_OPTIONAL_FUNC */
#define UCS_SYS_LIB_FUNC_LOAD_OPTIONAL(_prefix, _func, _lib_addr) \
    UCS_SYS_LIB_FUNC_PTR(_prefix, _func) = \
            UCS_SYS_LIB_OPTIONAL_FUNC(_lib_addr, _func)

END_C_DECLS

#endif /* LIB_H */
