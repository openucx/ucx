/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_PROFILE_ON_H_
#define UCS_PROFILE_ON_H_

#include "profile_defs.h"

#include <ucs/sys/compiler_def.h>
#include <ucs/sys/preprocessor.h>

BEGIN_C_DECLS

#define UCS_PROFILE_SAMPLE          UCS_PROFILE_SAMPLE_ALWAYS
#define UCS_PROFILE_CODE            UCS_PROFILE_CODE_ALWAYS
#define UCS_PROFILE_FUNC            UCS_PROFILE_FUNC_ALWAYS
#define UCS_PROFILE_FUNC_VOID       UCS_PROFILE_FUNC_VOID_ALWAYS
#define UCS_PROFILE_NAMED_CALL      UCS_PROFILE_NAMED_CALL_ALWAYS
#define UCS_PROFILE_CALL            UCS_PROFILE_CALL_ALWAYS
#define UCS_PROFILE_NAMED_CALL_VOID UCS_PROFILE_NAMED_CALL_VOID_ALWAYS
#define UCS_PROFILE_CALL_VOID       UCS_PROFILE_CALL_VOID_ALWAYS
#define UCS_PROFILE_REQUEST_EVENT   UCS_PROFILE_REQUEST_EVENT_ALWAYS


/*
 * Profile a new request allocation.
 *
 * @param _req      Request pointer.
 * @param _name     Allocation site name.
 * @param _param32  Custom 32-bit parameter.
 */
#define UCS_PROFILE_REQUEST_NEW(_req, _name, _param32) \
    UCS_PROFILE_CTX_RECORD_ALWAYS(ucs_profile_default_ctx, \
                                  UCS_PROFILE_TYPE_REQUEST_NEW, (_name), \
                                  (_param32), (uintptr_t)(_req));


/*
 * Profile a request progress event with status check.
 *
 * @param _req      Request pointer.
 * @param _name     Event name.
 * @param _param32  Custom 32-bit parameter.
 * @param _status   Status of the last progress event.
 */
#define UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(_req, _name, _param32, _status) \
    if (!UCS_STATUS_IS_ERR(_status)) { \
        UCS_PROFILE_REQUEST_EVENT((_req), (_name), (_param32)); \
    }


/*
 * Profile a request release.
 *
 * @param _req      Request pointer.
 */
#define UCS_PROFILE_REQUEST_FREE(_req) \
    UCS_PROFILE_CTX_RECORD_ALWAYS(ucs_profile_default_ctx, \
                                  UCS_PROFILE_TYPE_REQUEST_FREE, "", 0, \
                                  (uintptr_t)(_req));

END_C_DECLS

#endif
