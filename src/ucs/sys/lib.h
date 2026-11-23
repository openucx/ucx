/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
* Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
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
 * Find a symbol defined by the loaded library which contains an address.
 *
 * The lookup does not load any library, and a symbol which is defined only by
 * a dependency of that library is not returned. The caller must keep the
 * library loaded for as long as it uses the returned address.
 *
 * @param [in] lib_addr  Address inside the library, e.g. one of its functions.
 * @param [in] symbol    Name of the symbol to find.
 *
 * @return Address of the symbol in the requested library, or NULL if the
 *         lookup fails.
 */
void *ucs_sys_lib_dlsym(const void *lib_addr, const char *symbol);

END_C_DECLS

#endif /* LIB_H */
