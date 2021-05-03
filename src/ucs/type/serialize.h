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
 * Advance '_iter' by '_offset', and return a typed pointer of '_iter' before
 * it was advanced.
 *
 * @param _iter    Pointer to a pointer, representing the current position.
 * @param _type    Type of pointer to return (for example, 'const void' or
 *                 'uint8_t'). Passing the type explicitly helps to avoid
 *                 casting a const pointer to (non-const) void*.
 * @param _offset  Offset to advance the pointer _iter.
 *
 * @return '_iter' before it was advanced by '_offset', cast to '_type *'.
 */
#define ucs_serialize_next_raw(_iter, _type, _offset) \
    ({ \
        _type *_result = (_type*)(*(_iter)); \
        *(_iter)       = UCS_PTR_BYTE_OFFSET(*(_iter), _offset); \
        _result; \
    })


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
    ucs_serialize_next_raw(_iter, _type, sizeof(_type))

#endif
