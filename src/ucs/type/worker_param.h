/**
* Copyright (C) Mellanox Technologies Ltd. 20016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TYPE_CPU_PARAM_H
#define UCS_TYPE_CPU_PARAM_H

#include <sched.h>


/**
 * Specifies CPU parameters.
 */
typedef struct ucs_worker_param {
    cpu_set_t cpu_cores;
} ucs_worker_param_t;


#endif
