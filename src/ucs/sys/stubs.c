/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/stubs.h>
#include <ucs/debug/assert.h>


void ucs_empty_function(void)
{
}

unsigned ucs_empty_function_return_zero(void)
{
    return 0;
}

int64_t ucs_empty_function_return_zero_int64(void)
{
    return 0;
}

int ucs_empty_function_return_zero_int(void)
{
    return 0;
}

size_t ucs_empty_function_return_zero_size_t(void)
{
    return 0;
}

unsigned ucs_empty_function_return_one(void)
{
    return 1;
}

int ucs_empty_function_return_one_int(void)
{
    return 1;
}

ucs_status_t ucs_empty_function_return_success(void)
{
    return UCS_OK;
}

ucs_status_t ucs_empty_function_return_unsupported(void)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_ptr_t ucs_empty_function_return_ptr_unsupported(void)
{
    return UCS_STATUS_PTR(UCS_ERR_UNSUPPORTED);
}

ucs_status_t ucs_empty_function_return_inprogress(void)
{
    return UCS_INPROGRESS;
}

ucs_status_t ucs_empty_function_return_no_resource(void)
{
    return UCS_ERR_NO_RESOURCE;
}

ucs_status_t ucs_empty_function_return_invalid_param(void)
{
    return UCS_ERR_INVALID_PARAM;
}

ucs_status_ptr_t ucs_empty_function_return_ptr_no_resource(void)
{
    return UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE);
}

ucs_status_t ucs_empty_function_return_ep_timeout(void)
{
    return UCS_ERR_ENDPOINT_TIMEOUT;
}

ssize_t ucs_empty_function_return_bc_ep_timeout(void)
{
    return UCS_ERR_ENDPOINT_TIMEOUT;
}

ucs_status_t ucs_empty_function_return_busy(void)
{
    return UCS_ERR_BUSY;
}

int ucs_empty_function_do_assert(void)
{
    ucs_assert_always(0);
    return 0;
}

void ucs_empty_function_do_assert_void(void)
{
    ucs_assert_always(0);
}
