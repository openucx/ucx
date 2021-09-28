/**
* Copyright © 2021 NVIDIA CORPORATION & AFFILIATES.
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

END_C_DECLS

#endif /* LIB_H */
