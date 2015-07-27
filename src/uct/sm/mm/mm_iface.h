/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_MM_IFACE_H
#define UCT_MM_IFACE_H

#include <uct/tl/tl_base.h>
#include <ucs/debug/memtrack.h>
#include <sys/shm.h>
#include "mm_def.h"
#include "mm_pd.h"

#define UCT_MM_TL_NAME "mm"

#define UCT_MM_FIFO_CTL_SIZE_ALIGNED    ucs_align_up(sizeof(uct_mm_fifo_ctl_t),UCS_SYS_CACHE_LINE_SIZE)

typedef struct uct_mm_iface_config {
    uct_iface_config_t       super;
    unsigned                 fifo_size;            /* Size of the receive FIFO */
    double                   release_fifo_factor;
    ucs_ternary_value_t      hugetlb_mode;         /* Enable using huge pages for */
                                                   /* shared memory buffers */
    uct_iface_mpool_config_t mp;
} uct_mm_iface_config_t;


struct uct_mm_iface {
    uct_base_iface_t        super;

    /* Receive FIFO */
    uct_mm_id_t             fifo_mm_id;       /* memory id which will be received */
                                              /* after allocating the fifo */
    void                    *recv_fifo;       /* the beginning of the receive fifo */

    uct_mm_fifo_ctl_t       *recv_fifo_ctl;   /* pointer to the struct at the */
                                              /* beginning of the receive fifo */
                                              /* which holds the head and the tail */
    void                    *recv_fifo_elements; /* pointer to the first fifo element */
                                                 /* in the receive fifo */
    uint64_t                read_index;          /* actual reading location */

    uint8_t                 fifo_shift;          /* = log2(fifo_size) */
    unsigned                fifo_mask;           /* = 2^fifo_shift - 1 */
    unsigned                elem_size;
    uint64_t                fifo_release_factor_mask;

    ucs_mpool_h             recv_desc_mp;
    uct_mm_recv_desc_t      *last_recv_desc;

    struct {
        unsigned fifo_size;
        unsigned seg_size;          /* size of the receive descriptor */
                                    /* (for payload) */
    } config;
};


struct uct_mm_fifo_element {
    uint8_t             flags;
    uint8_t             am_id;     /* active message id */
    uint16_t            length;    /* length of actual data */
    /* the data follows here */
} UCS_S_PACKED;


struct uct_mm_fifo_ctl {
    volatile uint64_t  head;       /* where to write next */
    volatile uint64_t  tail;       /* how much was read */
} UCS_S_PACKED;


struct uct_mm_recv_desc {
    uct_mm_id_t         key;
    void                *base_address;
    uct_am_recv_desc_t  am_recv;   /* has to be in the end */
};


static UCS_F_ALWAYS_INLINE void
uct_mm_iface_invoke_am(uct_mm_iface_t *iface, uint8_t am_id, void *data,
                       unsigned length, uct_mm_recv_desc_t *mm_desc)
{
    ucs_status_t status;
    void *desc = mm_desc + 1;    /* point the desc to the user's headroom */

    status = uct_iface_invoke_am(&iface->super, am_id, data, length, desc);
    if (status != UCS_OK) {
        iface->last_recv_desc = ucs_mpool_get(iface->recv_desc_mp);
        if (iface->last_recv_desc == NULL ) {
           ucs_fatal("Failed to get a new receive descriptor for MM");
        }
        /* save the iface of this desc the its later release */
        uct_recv_desc_iface(desc) = &iface->super.super;
    }
}

void uct_mm_iface_release_am_desc(uct_iface_t *tl_iface, void *desc);
inline ucs_status_t uct_mm_flush();

extern uct_tl_component_t uct_mm_tl;

#endif
