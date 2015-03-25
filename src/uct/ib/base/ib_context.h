/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_IB_CONTEXT_H_
#define UCT_IB_CONTEXT_H_

#include "ib_device.h"
#include <ucs/sys/math.h>


enum {
    UCT_IB_RESOURCE_FLAG_MLX4_PRM = UCS_BIT(1),   /* Device supports mlx4 PRM */
    UCT_IB_RESOURCE_FLAG_MLX5_PRM = UCS_BIT(2),   /* Device supports mlx5 PRM */
    UCT_IB_RESOURCE_FLAG_DC       = UCS_BIT(3)    /* Device supports DC */
};

typedef struct uct_ib_context uct_ib_context_t;
struct uct_ib_context {
    unsigned                    num_devices;     /* Number of devices */
    uct_ib_device_t             **devices;       /* Array of devices */
};


/*
 * Helper function to list IB resources
 *
 * @param context          UCT context.
 * @param flags            Transport requirements from IB device (see UCT_IB_RESOURCE_FLAG_xx)
 * @param tl_hdr_len       How many bytes this transport adds on top of IB header (LRH+BTH+iCRC+vCRC)
 * @param tl_overhead_ns   How much overhead the transport adds to latency.
 * @param resources_p      Filled with a pointer to an array of resources.
 * @param num_resources_p  Filled with the number of resources.
 */
ucs_status_t uct_ib_query_resources(uct_context_h context, unsigned flags,
                                    size_t tl_hdr_len, uint64_t tl_overhead_ns,
                                    uct_resource_desc_t **resources_p,
                                    unsigned *num_resources_p);


#endif
