/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_MUSA_COPY_IFACE_H
#define UCT_MUSA_COPY_IFACE_H

#include <uct/musa/base/musa_base.h>
#include <uct/base/uct_iface.h>
#include <ucs/datastruct/mpool.h>

#define UCT_MUSA_COPY_TL_NAME "musa_copy"

typedef uint64_t uct_musa_copy_iface_addr_t;

typedef struct uct_musa_copy_flush_desc uct_musa_copy_flush_desc_t;

typedef struct {
    ucs_queue_elem_t          queue;
    musaEvent_t               event;
    uct_completion_t          *comp;
    uct_musa_copy_flush_desc_t *flush;
} uct_musa_copy_event_desc_t;

typedef struct {
    ucs_queue_elem_t  queue;
    musaStream_t      stream;
    ucs_queue_head_t  event_queue;
} uct_musa_copy_queue_desc_t;

typedef struct {
    int                              device;
    ucs_queue_head_t                 active_queue;
    uct_musa_copy_queue_desc_t       queue_desc[UCS_MEMORY_TYPE_LAST]
                                               [UCS_MEMORY_TYPE_LAST];
    ucs_mpool_t                      event_mp;
} uct_musa_copy_device_rsc_t;

struct uct_musa_copy_flush_desc {
    unsigned          count;
    uct_completion_t *comp;
    ucs_status_t      status;
};

typedef struct {
    uct_base_iface_t             super;
    uct_musa_copy_iface_addr_t   id;
    int                          device;
    unsigned                     num_devices;
    uct_musa_copy_device_rsc_t  *device_rscs;
    struct {
        unsigned max_poll;
        unsigned max_events;
        double   latency;
        double   bandwidth;
    } config;
} uct_musa_copy_iface_t;

typedef struct {
    uct_iface_config_t super;
    unsigned           max_poll;
    unsigned           max_events;
    double             latency;
    double             bandwidth;
} uct_musa_copy_iface_config_t;

ucs_status_t
uct_musa_copy_iface_flush(uct_iface_h tl_iface, unsigned flags,
                          uct_completion_t *comp);

unsigned uct_musa_copy_iface_progress(uct_iface_h tl_iface);

uct_musa_copy_device_rsc_t *
uct_musa_copy_iface_get_device_rsc(uct_musa_copy_iface_t *iface, int device);

ucs_status_t
uct_musa_copy_iface_get_stream(uct_musa_copy_device_rsc_t *device_rsc,
                               uct_musa_copy_queue_desc_t *queue_desc);

ucs_status_t uct_musa_copy_iface_post_event(
        uct_musa_copy_device_rsc_t *device_rsc,
        uct_musa_copy_queue_desc_t *queue_desc,
        uct_completion_t *comp,
        uct_musa_copy_flush_desc_t *flush);

#endif
