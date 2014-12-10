/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_IB_DEVICE_H
#define UCT_IB_DEVICE_H

#include "ib_verbs.h"

#include <uct/api/uct.h>
#include <ucs/type/status.h>


typedef struct uct_ib_device uct_ib_device_t;
struct uct_ib_device {
    uct_pd_t                    super;
    struct ibv_context          *ibv_context;    /* Verbs context */
    struct ibv_pd               *pd;             /* Protection domain */
    struct ibv_exp_device_attr  dev_attr;        /* Cached device attributes */
    uint8_t                     first_port;      /* Number of first port (usually 1) */
    uint8_t                     num_ports;       /* Amount of physical ports */
    cpu_set_t                   local_cpus;      /* CPUs local to device */
    pthread_t                   async_thread;    /* Async event thread */
    int                         stop_thread;
    struct ibv_exp_port_attr    port_attr[0];    /* Cached port attributes */
};


ucs_status_t uct_ib_device_create(uct_context_h context,
                                  struct ibv_device *ibv_device,
                                  uct_ib_device_t **dev_p);
void uct_ib_device_destroy(uct_ib_device_t *dev);


ucs_status_t uct_ib_device_port_check(uct_ib_device_t *dev, uint8_t port_num,
                                      unsigned flags);

ucs_status_t uct_ib_device_port_get_resource(uct_ib_device_t *dev, uint8_t port_num,
                                             uct_resource_desc_t *resource);

const char *uct_ib_device_name(uct_ib_device_t *dev);

ucs_status_t uct_ib_rkey_unpack(uct_context_h context, void *rkey_buffer,
                                uct_rkey_bundle_t *rkey_ob);

static inline struct ibv_exp_port_attr* uct_ib_device_port_attr(uct_ib_device_t *dev, uint8_t port_num)
{
    return &dev->port_attr[port_num - dev->first_port];
}

static inline struct ibv_mr* uct_ib_lkey_mr(uct_lkey_t lkey)
{
    return (struct ibv_mr *)(void*)lkey;
}

#endif
