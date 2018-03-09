/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_PROFILE_OFF_H_
#define UCS_PROFILE_OFF_H_

#include "profile_defs.h"

#include <ucs/sys/compiler_def.h>


#define UCS_PROFILE(...)                                    UCS_EMPTY_STATEMENT
#define UCS_PROFILE_SAMPLE(_name)                           UCS_EMPTY_STATEMENT
#define UCS_PROFILE_SCOPE_BEGIN()                           UCS_EMPTY_STATEMENT
#define UCS_PROFILE_SCOPE_END(_name)                        UCS_EMPTY_STATEMENT
#define UCS_PROFILE_CODE(_name)
#define UCS_PROFILE_FUNC(_ret_type, _name, _arglist, ...)   _ret_type _name(__VA_ARGS__)
#define UCS_PROFILE_FUNC_VOID(_name, _arglist, ...)         void _name(__VA_ARGS__)
#define UCS_PROFILE_NAMED_CALL(_name, _func, ...)           _func(__VA_ARGS__)
#define UCS_PROFILE_CALL(_func, ...)                        _func(__VA_ARGS__)
#define UCS_PROFILE_NAMED_CALL_VOID(_name, _func, ...)      _func(__VA_ARGS__)
#define UCS_PROFILE_CALL_VOID(_func, ...)                   _func(__VA_ARGS__)
#define UCS_PROFILE_REQUEST_NEW(...)                        UCS_EMPTY_STATEMENT
#define UCS_PROFILE_REQUEST_EVENT(...)                      UCS_EMPTY_STATEMENT
#define UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(...)         UCS_EMPTY_STATEMENT
#define UCS_PROFILE_REQUEST_FREE(...)                       UCS_EMPTY_STATEMENT

#endif
