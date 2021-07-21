/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/stubs.h>
#include <ucs/debug/assert.h>


void ucs_empty_function()
{
}

unsigned ucs_empty_function_return_zero()
{
    return 0;
}

int64_t ucs_empty_function_return_zero_int64()
{
    return 0;
}

int ucs_empty_function_return_zero_int()
{
    return 0;
}

size_t ucs_empty_function_return_zero_size_t()
{
    return 0;
}

unsigned ucs_empty_function_return_one()
{
    return 1;
}

int ucs_empty_function_return_one_int()
{
    return 1;
}

ucs_status_t ucs_empty_function_return_success()
{
    return UCS_OK;
}

ucs_status_t ucs_empty_function_return_unsupported()
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t ucs_empty_function_return_inprogress()
{
    return UCS_INPROGRESS;
}

ucs_status_t ucs_empty_function_return_no_resource()
{
    return UCS_ERR_NO_RESOURCE;
}

ucs_status_t ucs_empty_function_return_invalid_param()
{
    return UCS_ERR_INVALID_PARAM;
}

ucs_status_ptr_t ucs_empty_function_return_ptr_no_resource()
{
    return UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE);
}

ucs_status_t ucs_empty_function_return_ep_timeout()
{
    return UCS_ERR_ENDPOINT_TIMEOUT;
}

ssize_t ucs_empty_function_return_bc_ep_timeout()
{
    return UCS_ERR_ENDPOINT_TIMEOUT;
}

ucs_status_t ucs_empty_function_return_busy()
{
    return UCS_ERR_BUSY;
}

int ucs_empty_function_do_assert()
{
    ucs_assert_always(0);
    return 0;
}

void ucs_empty_function_do_assert_void()
{
    ucs_assert_always(0);
}
