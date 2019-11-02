/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_MM_IFACE_H
#define UCT_MM_IFACE_H

#include "mm_md.h"

#include <uct/base/uct_iface.h>
#include <uct/sm/base/sm_iface.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/memtrack.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
#include <sys/shm.h>
#include <sys/un.h>


enum {
    UCT_MM_FIFO_ELEM_FLAG_OWNER  = UCS_BIT(0), /* new/old info */
    UCT_MM_FIFO_ELEM_FLAG_INLINE = UCS_BIT(1), /* if inline or not */
};


#define UCT_MM_FIFO_CTL_SIZE_ALIGNED  ucs_align_up(sizeof(uct_mm_fifo_ctl_t),UCS_SYS_CACHE_LINE_SIZE)

#define UCT_MM_GET_FIFO_SIZE(iface)  (UCS_SYS_CACHE_LINE_SIZE - 1 +  \
                                      UCT_MM_FIFO_CTL_SIZE_ALIGNED + \
                                     ((iface)->config.fifo_size *    \
                                     (iface)->config.fifo_elem_size))

#define UCT_MM_IFACE_GET_FIFO_ELEM(_iface, _fifo , _index) \
          (uct_mm_fifo_element_t*) ((char*)(_fifo) + ((_index) * \
          (_iface)->config.fifo_elem_size))


#define UCT_MM_IFACE_GET_DESC_START(_iface, _fifo_elem_p) \
          ((uct_mm_recv_desc_t *)UCS_PTR_BYTE_OFFSET((_fifo_elem_p)->desc_chunk_base_addr,  \
          (_fifo_elem_p)->desc_offset - (_iface)->rx_headroom) - 1)


#define uct_mm_iface_mapper_call(_iface, _func, ...) \
    ({ \
        uct_mm_md_mapper_call((_iface)->super.super.md, _func, ## __VA_ARGS__); \
    })


/* Check if the resources on the remote peer are available for sending to it.
 * i.e. check if the remote receive FIFO has room in it.
 * return 1 if can send.
 * return 0 if can't send.
 */
#define UCT_MM_EP_IS_ABLE_TO_SEND(_head, _tail, _fifo_size) \
          ucs_likely(((_head) - (_tail)) < (_fifo_size))


/**
 * MM interface configuration
 */
typedef struct uct_mm_iface_config {
    uct_sm_iface_config_t    super;
    size_t                   seg_size;            /* Size of the receive
                                                   * descriptor (for payload) */
    unsigned                 fifo_size;           /* Size of the receive FIFO */
    double                   release_fifo_factor;
    ucs_ternary_value_t      hugetlb_mode;        /* Enable using huge pages for
                                                   * shared memory buffers */
    unsigned                 fifo_elem_size;      /* Size of the FIFO element size */
    uct_iface_mpool_config_t mp;
} uct_mm_iface_config_t;


/**
 * MM interface address
 */
typedef struct uct_mm_iface_addr {
    uint64_t   id;
    uintptr_t  vaddr;
} UCS_S_PACKED uct_mm_iface_addr_t;


/**
 * MM FIFO control segment
 */
typedef struct uct_mm_fifo_ctl {
    /* 1st cacheline */
    volatile uint64_t  head;       /* where to write next */
    socklen_t          signal_addrlen;   /* address length of signaling socket */
    struct sockaddr_un signal_sockaddr;  /* address of signaling socket */
    UCS_CACHELINE_PADDING(uint64_t, socklen_t, struct sockaddr_un);

    /* 2nd cacheline */
    volatile uint64_t  tail;       /* how much was read */
} UCS_S_PACKED UCS_V_ALIGNED(UCS_SYS_CACHE_LINE_SIZE) uct_mm_fifo_ctl_t;


/**
 * MM FIFO element
 */
typedef struct uct_mm_fifo_element {
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
} UCS_S_PACKED uct_mm_fifo_element_t;


/**
 * MM receive descriptor:
 */
typedef struct uct_mm_recv_desc {
    uct_mm_id_t         key;
    void                *base_address;
    size_t              mpool_length;
    uct_recv_desc_t     recv;   /* has to be in the end */
} uct_mm_recv_desc_t;


/**
 * MM trandport interface
 */
typedef struct uct_mm_iface {
    uct_sm_iface_t          super;

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
} uct_mm_iface_t;


/*
 * Define a memory-mapper transport for MM.
 *
 * @param _name         Component name token
 * @param _md_ops       Memory domain operations, of type uct_mm_md_ops_t.
 * @param _rkey_unpack  Remote key unpack function
 * @param _rkey_release Remote key release function
 * @param _cfg_prefix   Prefix for configuration variables.
 */
#define UCT_MM_TL_DEFINE(_name, _md_ops, _rkey_unpack, _rkey_release, \
                         _cfg_prefix) \
    \
    UCT_MM_COMPONENT_DEFINE(uct_##_name##_component, _name, _md_ops, \
                            _rkey_unpack, _rkey_release, _cfg_prefix) \
    \
    UCT_TL_DEFINE(&(uct_##_name##_component).super, \
                  _name, \
                  uct_sm_base_query_tl_devices, \
                  uct_mm_iface_t, \
                  "MM_", \
                  uct_mm_iface_config_table, \
                  uct_mm_iface_config_t);


extern ucs_config_field_t uct_mm_iface_config_table[];

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_mm_iface_invoke_am(uct_mm_iface_t *iface, uint8_t am_id, void *data,
                       unsigned length, unsigned flags)
{
    ucs_status_t status;
    void         *desc;

    status = uct_iface_invoke_am(&iface->super.super, am_id, data,
                                 length, flags);

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
   *fifo_elems = UCS_PTR_BYTE_OFFSET(fifo_ctl, UCT_MM_FIFO_CTL_SIZE_ALIGNED);
}

UCS_CLASS_DECLARE_NEW_FUNC(uct_mm_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                           const uct_iface_params_t*, const uct_iface_config_t*);

void uct_mm_iface_release_desc(uct_recv_desc_t *self, void *desc);

ucs_status_t uct_mm_flush();

unsigned uct_mm_iface_progress(void *arg);

#endif
