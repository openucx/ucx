/**
* Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_PARAM_H_
#define UCS_PARAM_H_

#include <ucs/sys/preprocessor.h>
#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS


/**
 * Conditionally return a param value, if a flag in the field mask is set.
 * Otherwise, return a default value.
 *
 * @param _prefix  Prefix of each value in the field mask enum
 * @param _params  Pointer to params struct
 * @param _name    Return this member of the params struct
 * @param _flag    Check for flag with this name
 * @param _default Return this value if the flag in the field mask is not set
 *
 * @return Param value (if the field mask flag is set) or the default value
 */
#define UCS_PARAM_VALUE(_prefix, _params, _name, _flag, _default) \
    (((_params)->field_mask & UCS_PP_TOKENPASTE3(_prefix, _, _flag)) ? \
             (_params)->_name : \
             (_default))


END_C_DECLS

#endif
