/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_MM_H
#define UCT_MM_H

#include <ucs/sys/math.h>
#include <ucs/datastruct/queue.h>

typedef struct uct_mm_ep                uct_mm_ep_t;
typedef struct uct_mm_iface             uct_mm_iface_t;
typedef struct uct_mm_fifo_ctl          uct_mm_fifo_ctl_t;
typedef struct uct_mm_fifo_element      uct_mm_fifo_element_t;
typedef struct uct_mm_recv_desc         uct_mm_recv_desc_t;

#define UCT_MM_FIFO_ELEMENT_SIZE  128

enum {
    UCT_MM_FIFO_ELEM_FLAG_OWNER  = UCS_BIT(0), /* new/old info */
    UCT_MM_FIFO_ELEM_FLAG_INLINE = UCS_BIT(1), /* if inline or not */
};

#define UCT_MM_IFACE_GET_FIFO_ELEM(_iface, _fifo , _index) \
    (uct_mm_fifo_element_t*) ((char*)(_fifo) + ((_index) * (_iface->elem_size)));

#endif /* UCT_MM_H */
