/**
* Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_SERIALIZE_H
#define UCS_SERIALIZE_H

#include <ucs/sys/compiler_def.h>


/*
 * Helper macro for serializing/deserializing custom data.
 * Advance '_iter' to the next element, and return a typed pointer to the
 * current element.
 *
 * @param _iter   Pointer to a pointer, representing the current element.
 * @param _type   Type of the current element.
 *
 * @return Typed pointer to the current element.
 */
#define ucs_serialize_next(_iter, _type) \
    ({ \
        _type *_result = (_type*)(*(_iter)); \
        *(_iter)       = UCS_PTR_TYPE_OFFSET(*(_iter), _type); \
        _result; \
    })

#endif
