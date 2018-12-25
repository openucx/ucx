/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_MM_IFACE_H
#define UCT_MM_IFACE_H

#include "mm_def.h"
#include "mm_md.h"

#include <uct/base/uct_iface.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/memtrack.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
#include <sys/shm.h>
#include <sys/un.h>


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


struct uct_mm_fifo_ctl {
    /* 1st cacheline */
    volatile uint64_t  head;       /* where to write next */
    socklen_t          signal_addrlen;   /* address length of signaling socket */
    struct sockaddr_un signal_sockaddr;  /* address of signaling socket */
    UCS_CACHELINE_PADDING(uint64_t, socklen_t, struct sockaddr_un);

    /* 2nd cacheline */
    volatile uint64_t  tail;       /* how much was read */
} UCS_S_PACKED;


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

    int                     signal_fd;        /* Unix socket for receiving remote signal */

    size_t                  rx_headroom;
    ucs_arbiter_t           arbiter;
    const char              *path;            /* path to the backing file (for 'posix') */
    uct_recv_desc_t         release_desc;

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


struct uct_mm_recv_desc {
    uct_mm_id_t         key;
    void                *base_address;
    size_t              mpool_length;
    uct_recv_desc_t     recv;   /* has to be in the end */
};


static UCS_F_ALWAYS_INLINE ucs_status_t
uct_mm_iface_invoke_am(uct_mm_iface_t *iface, uint8_t am_id, void *data,
                       unsigned length, unsigned flags)
{
    ucs_status_t status;
    void         *desc;

    status = uct_iface_invoke_am(&iface->super, am_id, data, length, flags);

    if (status == UCS_INPROGRESS) {
        desc = (void *)((uintptr_t)data - iface->rx_headroom);
        /* save the release_desc for later release of this desc */
        uct_recv_desc(desc) = &iface->release_desc;
    }

    return status;
}


static uct_mm_fifo_ctl_t* uct_mm_set_fifo_ctl(void *mem_region)
{
    return (uct_mm_fifo_ctl_t*) ucs_align_up_pow2
           ((uintptr_t) mem_region , UCS_SYS_CACHE_LINE_SIZE);
}

/**
 * Set aligned pointers of the FIFO according to the beginning of the allocated
 * memory.
 *
 * @param [in] mem_region  pointer to the beginning of the allocated memory.
 * @param [out] fifo_elems an aligned pointer to the first FIFO element.
 */
static inline void uct_mm_set_fifo_elems_ptr(void *mem_region, void **fifo_elems)
{
   uct_mm_fifo_ctl_t *fifo_ctl;

   /* initiate the the uct_mm_fifo_ctl struct, holding the head and the tail */
   fifo_ctl = uct_mm_set_fifo_ctl(mem_region);

   /* initiate the pointer to the beginning of the first FIFO element */
   *fifo_elems = (void*) fifo_ctl + UCT_MM_FIFO_CTL_SIZE_ALIGNED;
}

void uct_mm_iface_release_desc(uct_recv_desc_t *self, void *desc);
ucs_status_t uct_mm_flush();

unsigned uct_mm_iface_progress(void *arg);

extern uct_tl_component_t uct_mm_tl;

#endif
