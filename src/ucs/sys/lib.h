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
 * Look up an optional symbol in the already loaded library which contains
 * @a lib_addr. The symbol is returned only if that library defines it, and
 * the caller must keep the library loaded while using it.
 *
 * @param [in] lib_addr     Address of any symbol of the library to search.
 * @param [in] sym_name     Name of the optional symbol.
 * @param [in] sym_version  Symbol version node, or NULL for the default
 *                          version. Ignored if dlvsym() is not available.
 *
 * @return Symbol address, or NULL if the library does not define it.
 */
void *ucs_sys_get_lib_sym(const void *lib_addr, const char *sym_name,
                          const char *sym_version);


/**
 * Get an optional function defined by the library which contains
 * @a _lib_addr. Shared libraries look it up at runtime, so loading them does
 * not require @a _version; static archives reference it directly.
 *
 * @param [in] _lib_addr  Function linked from the same library.
 * @param [in] _func      Optional function declared by the included headers.
 * @param [in] _version   Version node that introduced the declared prototype.
 *
 * @return Pointer to @a _func, or NULL if the loaded library does not define
 *         it.
 */
#ifdef UCX_SHARED_LIB
#  define UCS_SYS_LIB_OPTIONAL_FUNC(_lib_addr, _func, _version) \
      ((ucs_typeof(_func)*)ucs_sys_get_lib_sym((const void*)(_lib_addr), \
                                               #_func, _version))
#else
#  define UCS_SYS_LIB_OPTIONAL_FUNC(_lib_addr, _func, _version) (&(_func))
#endif

END_C_DECLS

#endif /* LIB_H */
