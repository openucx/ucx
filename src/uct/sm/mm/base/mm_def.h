/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_MM_H
#define UCT_MM_H

#include <uct/base/uct_md.h>
#include <ucs/sys/math.h>
#include <ucs/datastruct/queue.h>
#include <uct/api/uct_def.h>


typedef struct uct_mm_ep                uct_mm_ep_t;
typedef struct uct_mm_iface             uct_mm_iface_t;
typedef struct uct_mm_fifo_ctl          uct_mm_fifo_ctl_t;
typedef struct uct_mm_fifo_element      uct_mm_fifo_element_t;
typedef struct uct_mm_recv_desc         uct_mm_recv_desc_t;
typedef struct uct_mm_remote_seg        uct_mm_remote_seg_t;

#define UCT_MM_BASE_ADDRESS_HASH_SIZE    64

enum {
    UCT_MM_FIFO_ELEM_FLAG_OWNER  = UCS_BIT(0), /* new/old info */
    UCT_MM_FIFO_ELEM_FLAG_INLINE = UCS_BIT(1), /* if inline or not */
};

enum {
    UCT_MM_AM_BCOPY,
    UCT_MM_AM_SHORT,
};

#define UCT_MM_IFACE_GET_FIFO_ELEM(_iface, _fifo , _index) \
          (uct_mm_fifo_element_t*) ((char*)(_fifo) + ((_index) * \
          (_iface)->config.fifo_elem_size));

#define UCT_MM_IFACE_GET_DESC_START(_iface, _fifo_elem_p) \
          (uct_mm_recv_desc_t *) ((_fifo_elem_p)->desc_chunk_base_addr +  \
          (_fifo_elem_p)->desc_offset - (_iface)->rx_headroom) - 1;


/* Check if the resources on the remote peer are available for sending to it.
 * i.e. check if the remote receive FIFO has room in it.
 * return 1 if can send.
 * return 0 if can't send.
 */
#define UCT_MM_EP_IS_ABLE_TO_SEND(_head, _tail, _fifo_size) \
          ucs_likely(((_head) - (_tail)) < (_fifo_size))

typedef struct uct_mm_md_config {
    uct_md_config_t      super;
    ucs_ternary_value_t  hugetlb_mode;     /* Enable using huge pages */
} uct_mm_md_config_t;


typedef struct uct_mm_iface_addr {
    uint64_t   id;
    uintptr_t  vaddr;
} UCS_S_PACKED uct_mm_iface_addr_t;


#endif /* UCT_MM_H */
