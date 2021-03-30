/**
* Copyright (C) NVIDIA Corporation. 2021.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCS_PROFILE_RANGE_H_
#define UCS_PROFILE_RANGE_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/compiler_def.h>
#include <stdint.h>

BEGIN_C_DECLS

/** @file profile_range.h */

/*
 * Start a range trace on an arbitrary event in a potentially nested fashion.
 * A range tracing can be started in a function and potentially end in an
 * another function or a recursive invocation of the same function.
 * A unique ID is returned to stop tracing the event.
 *
 * @param [in]     format      String name for the range.
 *
 * @return ID to be used to stop tracing a range
 */
uint64_t ucs_profile_range_start(const char *format, ...);


/*
 * Stop range trace.
 *
 * @param [in]     id          id that was returned from range start.
 *
 */
void ucs_profile_range_stop(uint64_t id);


/*
 * Add a marker on trace profiles.
 *
 * @param [in]     format      String name for the marker.
 *
 */
void ucs_profile_range_add_marker(const char *format, ...);


/*
 * Start a range trace in a non-nested fashion. Range tracing must start and end
 * in the same function.
 *
 * @param [in]     format      String name for the marker.
 *
 */
void ucs_profile_range_push(const char *format, ...);


/*
 * Stop a range trace in a non-nested fashion.
 *
 */
void ucs_profile_range_pop();


END_C_DECLS

#endif
