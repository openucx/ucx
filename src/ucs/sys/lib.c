/**
* Copyright © 2021 NVIDIA CORPORATION & AFFILIATES.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "lib.h"


ucs_status_t ucs_sys_get_lib_info(Dl_info *dl_info)
{
    int ret;

    (void)dlerror();
    ret = dladdr(ucs_sys_get_lib_info, dl_info);
    if (ret == 0) {
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

const char *ucs_sys_get_lib_path()
{
    ucs_status_t status;
    Dl_info dl_info;

    status = ucs_sys_get_lib_info(&dl_info);
    if (status != UCS_OK) {
        return "<failed to resolve libucs path>";
    }

    return dl_info.dli_fname;
}

unsigned long ucs_sys_get_lib_base_addr()
{
    ucs_status_t status;
    Dl_info dl_info;

    status = ucs_sys_get_lib_info(&dl_info);
    if (status != UCS_OK) {
        return 0;
    }

    return (uintptr_t)dl_info.dli_fbase;
}
