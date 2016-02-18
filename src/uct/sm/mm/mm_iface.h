/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_MM_IFACE_H
#define UCT_MM_IFACE_H

#include "mm_def.h"
#include "mm_pd.h"

#include <uct/base/uct_iface.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/memtrack.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/sys/sys.h>
#include <sys/shm.h>


#define UCT_MM_TL_NAME "mm"
#define UCT_MM_FIFO_CTL_SIZE_ALIGNED  ucs_align_up(sizeof(uct_mm_fifo_ctl_t),UCS_SYS_CACHE_LINE_SIZE)

#define UCT_MM_GET_FIFO_SIZE(iface)  (UCS_SYS_CACHE_LINE_SIZE - 1 +  \
                                      UCT_MM_FIFO_CTL_SIZE_ALIGNED + \
                                     ((iface)->config.fifo_size *    \
                                     (iface)->config.fifo_elem_size))

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
    void                    *shared_mem;      /* the beginning of the receive fifo */

    uct_mm_fifo_ctl_t       *recv_fifo_ctl;   /* pointer to the struct at the */
                                              /* beginning of the receive fifo */
                                              /* which holds the head and the tail. */
                                              /* this struct is cache line aligned and */
                                              /* doesn't necessarily start where */
                                              /* shared_mem starts */
    void                    *recv_fifo_elements; /* pointer to the first fifo element */
                                                 /* in the receive fifo */
    uint64_t                read_index;          /* actual reading location */

    uint8_t                 fifo_shift;          /* = log2(fifo_size) */
    unsigned                fifo_mask;           /* = 2^fifo_shift - 1 */
    uint64_t                fifo_release_factor_mask;

    ucs_mpool_t             recv_desc_mp;
    uct_mm_recv_desc_t      *last_recv_desc;    /* next receive descriptor to use */

    size_t                  rx_headroom;
    ucs_arbiter_t           arbiter;
    const char              *path;            /* path to the backing file (for 'posix') */

    struct {
        unsigned fifo_size;
        unsigned fifo_elem_size;
        unsigned seg_size;                    /* size of the receive descriptor (for payload)*/
    } config;
};


struct uct_mm_fifo_element {
    uint8_t         flags;
    uint8_t         am_id;          /* active message id */
    uint16_t        length;         /* length of actual data */

    /* bcopy parameters */
    size_t          desc_mpool_size;
    uct_mm_id_t     desc_mmid;      /* the mmid of the the memory chunk that
                                     * the desc (that this fifo_elem points to)
                                     * belongs to */
    size_t          desc_offset;    /* the offset of the desc (its data location for bcopy)
                                     * within the memory chunk it belongs to */
    void            *desc_chunk_base_addr;
    /* the data follows here (in case of inline messaging) */
} UCS_S_PACKED;


struct uct_mm_fifo_ctl {
    volatile uint64_t  head;       /* where to write next */
    volatile uint64_t  tail;       /* how much was read */
} UCS_S_PACKED;


struct uct_mm_recv_desc {
    uct_mm_id_t         key;
    void                *base_address;
    size_t              mpool_length;
    uct_am_recv_desc_t  am_recv;   /* has to be in the end */
};


static UCS_F_ALWAYS_INLINE ucs_status_t
uct_mm_iface_invoke_am(uct_mm_iface_t *iface, uint8_t am_id, void *data,
                       unsigned length, uct_mm_recv_desc_t *mm_desc)
{
    ucs_status_t status;
    void *desc = mm_desc + 1;    /* point the desc to the user's headroom */

    status = uct_iface_invoke_am(&iface->super, am_id, data, length, desc);
    if (status != UCS_OK) {
        /* save the iface of this desc for its later release */
        uct_recv_desc_iface(desc) = &iface->super.super;
    }
    return status;
}

/**
 * Set aligned pointers of the FIFO according to the beginning of the allocated
 * memory.
 *
 * @param [in] mem_region  pointer to the beginning of the allocated memory.
 * @param [out] fifo_ctl an aligned pointer to the beginning of the ctl struct in the FIFO
 * @param [out] fifo_elems an aligned pointer to the first FIFO element.
 */
static inline void uct_mm_set_fifo_ptrs(void *mem_region, uct_mm_fifo_ctl_t **fifo_ctl,
		                                void **fifo_elems)
{
   /* initiate the the uct_mm_fifo_ctl struct, holding the head and the tail */
   *fifo_ctl   = (uct_mm_fifo_ctl_t*) ucs_align_up_pow2
                ((uintptr_t) mem_region , UCS_SYS_CACHE_LINE_SIZE);

   /* initiate the pointer to the beginning of the first FIFO element */
   *fifo_elems = (void*) *fifo_ctl + UCT_MM_FIFO_CTL_SIZE_ALIGNED;
}

void uct_mm_iface_release_am_desc(uct_iface_t *tl_iface, void *desc);
ucs_status_t uct_mm_flush();

extern uct_tl_component_t uct_mm_tl;

#endif
