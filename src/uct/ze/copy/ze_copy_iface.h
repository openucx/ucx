/*
 * Copyright (C) Intel Corporation, 2023-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_ZE_COPY_IFACE_H
#define UCT_ZE_COPY_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/datastruct/array.h>
#include <level_zero/ze_api.h>


#define UCT_ZE_COPY_TL_NAME "ze_cpy"


typedef uint64_t uct_ze_copy_iface_addr_t;


/* Command queue and list which run copies on one device */
typedef struct {
    ze_device_handle_t        device;
    ze_command_queue_handle_t cmdq;
    ze_command_list_handle_t  cmdl;
} uct_ze_copy_queue_t;


UCS_ARRAY_DECLARE_TYPE(uct_ze_copy_queue_array_t, unsigned,
                       uct_ze_copy_queue_t);


typedef struct uct_ze_copy_iface {
    uct_base_iface_t          super;
    uct_ze_copy_iface_addr_t  id;
    /* Element 0 is on the MD device, the rest are created on first use.
     * Growth may move the elements and nothing here locks, so like the
     * single command list before it this is not safe for concurrent use of
     * one iface (UCS_THREAD_MODE_MULTI); UCP opens UCT workers in SINGLE or
     * SERIALIZED mode only. */
    uct_ze_copy_queue_array_t queues;
} uct_ze_copy_iface_t;


/* Return the index of the queue to copy from src to dst; the index stays
 * valid for the iface lifetime, an element pointer may not. */
ucs_status_t uct_ze_copy_iface_get_queue(uct_ze_copy_iface_t *iface,
                                         const void *src, const void *dst,
                                         unsigned *queue_index_p);

#endif
