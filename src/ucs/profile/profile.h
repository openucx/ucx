/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_PROFILE_H_
#define UCS_PROFILE_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#ifdef HAVE_PROFILING
#  include "profile_on.h"
#else
#  include "profile_off.h"
#endif

void ucs_profile_reset_locations_id(ucs_profile_context_t *ctx);

#endif
