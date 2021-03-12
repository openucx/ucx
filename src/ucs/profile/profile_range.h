/**
* Copyright (C) NVIDIA Corporation. 2021.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCS_PROFILE_RANGE_H_
#define UCS_PROFILE_RANGE_H_

BEGIN_C_DECLS

/** @file profile_range.h */

/*
 * Store a new record with the given data.
 * SHOULD NOT be used directly - use UCS_PROFILE macros instead.
 *
 * @param [in]     type        Location type.
 * @param [in]     name        Location name.
 * @param [in]     param32     custom 32-bit parameter.
 * @param [in]     param64     custom 64-bit parameter.
 * @param [in]     file        Source file name.
 * @param [in]     line        Source line number.
 * @param [in]     function    Calling function name.
 * @param [in,out] loc_id_p    Variable used to maintain the location ID.
 */

uint64_t ucs_profile_range_start(const char *format, ...);

void ucs_profile_range_stop(uint64_t id);

void ucs_profile_range_add_marker(const char *format, ...);

void ucs_profile_range_push(const char *format, ...);

void ucs_profile_range_pop();


END_C_DECLS

#endif
