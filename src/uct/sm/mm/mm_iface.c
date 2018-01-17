/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "mm_iface.h"
#include "mm_ep.h"

#include <uct/base/uct_worker.h>
#include <uct/sm/base/sm_ep.h>
#include <uct/sm/base/sm_iface.h>
#include <ucs/arch/atomic.h>
#include <ucs/arch/bitops.h>
#include <ucs/async/async.h>
#include <ucs/sys/string.h>
#include <sys/poll.h>


/* Maximal number of events to clear from the signaling pipe in single call */
#define UCT_MM_IFACE_MAX_SIG_EVENTS  32


static ucs_config_field_t uct_mm_iface_config_table[] = {
    {"", "ALLOC=md", NULL,
     ucs_offsetof(uct_mm_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"FIFO_SIZE", "64",
     "Size of the receive FIFO in the memory-map UCTs.",
     ucs_offsetof(uct_mm_iface_config_t, fifo_size), UCS_CONFIG_TYPE_UINT},

    {"FIFO_RELEASE_FACTOR", "0.5",
     "Frequency of resource releasing on the receiver's side in the MM UCT.\n"
     "This value refers to the percentage of the FIFO size. (must be >= 0 and < 1)",
     ucs_offsetof(uct_mm_iface_config_t, release_fifo_factor), UCS_CONFIG_TYPE_DOUBLE},

    UCT_IFACE_MPOOL_CONFIG_FIELDS("RX_", -1, 512, "receive",
                                  ucs_offsetof(uct_mm_iface_config_t, mp), ""),

    {"FIFO_HUGETLB", "no",
     "Enable using huge pages for internal shared memory buffers."
     "Possible values are:\n"
     " y   - Allocate memory using huge pages only.\n"
     " n   - Allocate memory using regular pages only.\n"
     " try - Try to allocate memory using huge pages and if it fails, allocate regular pages.\n",
     ucs_offsetof(uct_mm_iface_config_t, hugetlb_mode), UCS_CONFIG_TYPE_TERNARY},

    {NULL}
};

static ucs_status_t uct_mm_iface_get_address(uct_iface_t *tl_iface,
                                             uct_iface_addr_t *addr)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_iface, uct_mm_iface_t);
    uct_mm_iface_addr_t *iface_addr = (void*)addr;

    iface_addr->id    = iface->fifo_mm_id;
    iface_addr->vaddr = (uintptr_t)iface->shared_mem;
    return UCS_OK;
}

void uct_mm_iface_release_desc(uct_recv_desc_t *self, void *desc)
{
    void *mm_desc;

    mm_desc = desc - sizeof(uct_mm_recv_desc_t);
    ucs_mpool_put(mm_desc);
}

ucs_status_t uct_mm_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                uct_completion_t *comp)
{
    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    ucs_memory_cpu_store_fence();
    UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_OK;
}

static ucs_status_t uct_mm_iface_query(uct_iface_h tl_iface,
                                       uct_iface_attr_t *iface_attr)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_iface, uct_mm_iface_t);
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    /* default values for all shared memory transports */
    iface_attr->cap.put.max_short       = UINT_MAX;
    iface_attr->cap.put.max_bcopy       = SIZE_MAX;
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = SIZE_MAX;
    iface_attr->cap.put.opt_zcopy_align = UCS_SYS_CACHE_LINE_SIZE;
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.max_bcopy       = SIZE_MAX;
    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = SIZE_MAX;
    iface_attr->cap.get.opt_zcopy_align = UCS_SYS_CACHE_LINE_SIZE;
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = 1;

    iface_attr->cap.am.max_short        = iface->config.fifo_elem_size -
                                          sizeof(uct_mm_fifo_element_t);
    iface_attr->cap.am.max_bcopy        = iface->config.seg_size;
    iface_attr->cap.am.min_zcopy        = 0;
    iface_attr->cap.am.max_zcopy        = 0;
    iface_attr->cap.am.opt_zcopy_align  = UCS_SYS_CACHE_LINE_SIZE;
    iface_attr->cap.am.align_mtu        = iface_attr->cap.am.opt_zcopy_align;
    iface_attr->cap.am.max_iov          = 1;

    iface_attr->iface_addr_len          = sizeof(uct_mm_iface_addr_t);
    iface_attr->device_addr_len         = UCT_SM_IFACE_DEVICE_ADDR_LEN;
    iface_attr->ep_addr_len             = 0;
    iface_attr->max_conn_priv           = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_PUT_SHORT           |
                                          UCT_IFACE_FLAG_PUT_BCOPY           |
                                          UCT_IFACE_FLAG_ATOMIC_ADD32        |
                                          UCT_IFACE_FLAG_ATOMIC_ADD64        |
                                          UCT_IFACE_FLAG_ATOMIC_FADD64       |
                                          UCT_IFACE_FLAG_ATOMIC_FADD32       |
                                          UCT_IFACE_FLAG_ATOMIC_SWAP64       |
                                          UCT_IFACE_FLAG_ATOMIC_SWAP32       |
                                          UCT_IFACE_FLAG_ATOMIC_CSWAP64      |
                                          UCT_IFACE_FLAG_ATOMIC_CSWAP32      |
                                          UCT_IFACE_FLAG_ATOMIC_CPU          |
                                          UCT_IFACE_FLAG_GET_BCOPY           |
                                          UCT_IFACE_FLAG_AM_SHORT            |
                                          UCT_IFACE_FLAG_AM_BCOPY            |
                                          UCT_IFACE_FLAG_PENDING             |
                                          UCT_IFACE_FLAG_CB_SYNC             |
                                          UCT_IFACE_FLAG_EVENT_SEND_COMP     |
                                          UCT_IFACE_FLAG_EVENT_RECV_SIG      |
                                          UCT_IFACE_FLAG_CONNECT_TO_IFACE;

    iface_attr->latency.overhead        = 80e-9; /* 80 ns */
    iface_attr->latency.growth          = 0;
    iface_attr->bandwidth               = 6911 * 1024.0 * 1024.0;
    iface_attr->overhead                = 10e-9; /* 10 ns */
    iface_attr->priority                = uct_mm_md_mapper_ops(iface->super.md)->get_priority();
    return UCS_OK;
}

static inline void uct_mm_progress_fifo_tail(uct_mm_iface_t *iface)
{
    /* don't progress the tail every time - release in batches. improves performance */
    if (iface->read_index & iface->fifo_release_factor_mask) {
        return;
    }

    iface->recv_fifo_ctl->tail = iface->read_index;
}

ucs_status_t uct_mm_assign_desc_to_fifo_elem(uct_mm_iface_t *iface,
                                             uct_mm_fifo_element_t *fifo_elem_p,
                                             unsigned need_new_desc)
{
    uct_mm_recv_desc_t *desc;

    if (!need_new_desc) {
        desc = iface->last_recv_desc;
    } else {
        UCT_TL_IFACE_GET_RX_DESC(&iface->super, &iface->recv_desc_mp, desc,
                                 return UCS_ERR_NO_RESOURCE);
    }

    fifo_elem_p->desc_mmid   = desc->key;
    fifo_elem_p->desc_offset = iface->rx_headroom +
                               (ptrdiff_t) ((void*) (desc + 1) - desc->base_address);
    fifo_elem_p->desc_chunk_base_addr = desc->base_address;
    fifo_elem_p->desc_mpool_size      = desc->mpool_length;

    return UCS_OK;
}

static inline ucs_status_t uct_mm_iface_process_recv(uct_mm_iface_t *iface,
                                                     uct_mm_fifo_element_t* elem)
{
    ucs_status_t status;
    void         *data;

    if (ucs_likely(elem->flags & UCT_MM_FIFO_ELEM_FLAG_INLINE)) {
        /* read short (inline) messages from the FIFO elements */
        uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_RECV, elem->am_id,
                           elem + 1, elem->length, "RX: AM_SHORT");
        status = uct_mm_iface_invoke_am(iface, elem->am_id, elem + 1,
                                        elem->length, 0);
    } else {
        /* read bcopy messages from the receive descriptors */
        VALGRIND_MAKE_MEM_DEFINED(elem->desc_chunk_base_addr + elem->desc_offset,
                                  elem->length);

        data = elem->desc_chunk_base_addr + elem->desc_offset;

        uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_RECV, elem->am_id,
                           data, elem->length, "RX: AM_BCOPY");

        status = uct_mm_iface_invoke_am(iface, elem->am_id, data, elem->length,
                                        UCT_CB_PARAM_FLAG_DESC);
        if (status != UCS_OK) {
            /* assign a new receive descriptor to this FIFO element.*/
            uct_mm_assign_desc_to_fifo_elem(iface, elem, 0);
        }
    }
    return status;
}

static inline unsigned uct_mm_iface_poll_fifo(uct_mm_iface_t *iface)
{
    uint64_t read_index_loc, read_index;
    uct_mm_fifo_element_t* read_index_elem;
    ucs_status_t status;

    /* check the memory pool to make sure that there is a new descriptor available */
    if (ucs_unlikely(iface->last_recv_desc == NULL)) {
        UCT_TL_IFACE_GET_RX_DESC(&iface->super, &iface->recv_desc_mp,
                                 iface->last_recv_desc, return 0);
    }

    read_index = iface->read_index;
    read_index_loc = (read_index & iface->fifo_mask);
    /* the fifo_element which the read_index points to */
    read_index_elem = UCT_MM_IFACE_GET_FIFO_ELEM(iface, iface->recv_fifo_elements ,read_index_loc);

    /* check the read_index to see if there is a new item to read (checking the owner bit) */
    if (((read_index >> iface->fifo_shift) & 1) == ((read_index_elem->flags) & 1)) {

        /* read from read_index_elem */
        ucs_memory_cpu_load_fence();
        ucs_assert(iface->read_index <= iface->recv_fifo_ctl->head);

        status = uct_mm_iface_process_recv(iface, read_index_elem);
        if (status != UCS_OK) {
            /* the last_recv_desc is in use. get a new descriptor for it */
            UCT_TL_IFACE_GET_RX_DESC(&iface->super, &iface->recv_desc_mp,
                                     iface->last_recv_desc, ucs_debug("recv mpool is empty"));
        }

        /* raise the read_index. */
        iface->read_index++;

        uct_mm_progress_fifo_tail(iface);

        return 1;
    } else {
        return 0;
    }
}

unsigned uct_mm_iface_progress(void *arg)
{
    uct_mm_iface_t *iface = arg;
    unsigned count;

    /* progress receive */
    count = uct_mm_iface_poll_fifo(iface);

    /* progress the pending sends (if there are any) */
    ucs_arbiter_dispatch(&iface->arbiter, 1, uct_mm_ep_process_pending, NULL);

    return count;
}

static ucs_status_t uct_mm_iface_event_fd_get(uct_iface_h tl_iface, int *fd_p)
{
    *fd_p = ucs_derived_of(tl_iface, uct_mm_iface_t)->signal_fd;
    return UCS_OK;
}

static ucs_status_t uct_mm_iface_event_fd_arm(uct_iface_h tl_iface,
                                              unsigned events)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_iface, uct_mm_iface_t);
    char dummy[UCT_MM_IFACE_MAX_SIG_EVENTS]; /* pop multiple signals at once */
    int ret;

    ret = recvfrom(iface->signal_fd, &dummy, sizeof(dummy), 0, NULL, 0);
    if (ret > 0) {
        return UCS_ERR_BUSY;
    } else if (ret == -1) {
        if (errno == EAGAIN) {
            return UCS_OK;
        } else if (errno == EINTR) {
            return UCS_ERR_BUSY;
        } else {
            ucs_error("failed to retrieve message from signal pipe: %m");
            return UCS_ERR_IO_ERROR;
        }
    } else {
        ucs_assert(ret == 0);
        return UCS_OK;
    }
}

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_mm_iface_t, uct_iface_t);

static uct_iface_ops_t uct_mm_iface_ops = {
    .ep_put_short             = uct_sm_ep_put_short,
    .ep_put_bcopy             = uct_sm_ep_put_bcopy,
    .ep_get_bcopy             = uct_sm_ep_get_bcopy,
    .ep_am_short              = uct_mm_ep_am_short,
    .ep_am_bcopy              = uct_mm_ep_am_bcopy,
    .ep_atomic_add64          = uct_sm_ep_atomic_add64,
    .ep_atomic_fadd64         = uct_sm_ep_atomic_fadd64,
    .ep_atomic_cswap64        = uct_sm_ep_atomic_cswap64,
    .ep_atomic_swap64         = uct_sm_ep_atomic_swap64,
    .ep_atomic_add32          = uct_sm_ep_atomic_add32,
    .ep_atomic_fadd32         = uct_sm_ep_atomic_fadd32,
    .ep_atomic_cswap32        = uct_sm_ep_atomic_cswap32,
    .ep_atomic_swap32         = uct_sm_ep_atomic_swap32,
    .ep_pending_add           = uct_mm_ep_pending_add,
    .ep_pending_purge         = uct_mm_ep_pending_purge,
    .ep_flush                 = uct_mm_ep_flush,
    .ep_fence                 = uct_sm_ep_fence,
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_mm_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_mm_ep_t),
    .iface_flush              = uct_mm_iface_flush,
    .iface_fence              = uct_sm_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = (void*)uct_mm_iface_progress,
    .iface_event_fd_get       = uct_mm_iface_event_fd_get,
    .iface_event_arm          = uct_mm_iface_event_fd_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_mm_iface_t),
    .iface_query              = uct_mm_iface_query,
    .iface_get_device_address = uct_sm_iface_get_device_address,
    .iface_get_address        = uct_mm_iface_get_address,
    .iface_is_reachable       = uct_sm_iface_is_reachable
};

void uct_mm_iface_recv_desc_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh)
{
    uct_mm_recv_desc_t *desc = obj;
    uct_mm_seg_t *seg = memh;

    /* every desc in the memory pool, holds the mm_id(key) and address of the
     * mem pool it belongs to */
    desc->key          = seg->mmid;
    desc->base_address = seg->address;
    desc->mpool_length = seg->length;
}

static void uct_mm_iface_free_rx_descs(uct_mm_iface_t *iface, unsigned num_elems)
{
    uct_mm_fifo_element_t* fifo_elem_p;
    uct_mm_recv_desc_t *desc;
    unsigned i;

    for (i = 0; i < num_elems; i++) {
        fifo_elem_p = UCT_MM_IFACE_GET_FIFO_ELEM(iface, iface->recv_fifo_elements, i);
        desc = UCT_MM_IFACE_GET_DESC_START(iface, fifo_elem_p);
        ucs_mpool_put(desc);
    }
}

ucs_status_t uct_mm_allocate_fifo_mem(uct_mm_iface_t *iface,
                                      uct_mm_iface_config_t *config, uct_md_h md)
{
    uct_mm_fifo_ctl_t *ctl;
    size_t size_to_alloc;
    ucs_status_t status;

    /* allocate the receive FIFO */
    size_to_alloc = UCT_MM_GET_FIFO_SIZE(iface);

    status = uct_mm_md_mapper_ops(md)->alloc(md, &size_to_alloc, config->hugetlb_mode,
                                             0, &iface->shared_mem, &iface->fifo_mm_id,
                                             &iface->path UCS_MEMTRACK_NAME("mm fifo"));
    if (status != UCS_OK) {
        ucs_error("Failed to allocate memory for the receive FIFO in mm. size: %zu : %m",
                   size_to_alloc);
        return status;
    }

    ctl = uct_mm_set_fifo_ctl(iface->shared_mem);
    uct_mm_set_fifo_elems_ptr(iface->shared_mem, &iface->recv_fifo_elements);

    /* Make sure head and tail are cache-aligned, and not on same cacheline, to
     * avoid false-sharing.
     */
    ucs_assert_always((((uintptr_t)&ctl->head) % UCS_SYS_CACHE_LINE_SIZE) == 0);
    ucs_assert_always((((uintptr_t)&ctl->tail) % UCS_SYS_CACHE_LINE_SIZE) == 0);
    ucs_assert_always(((uintptr_t)&ctl->tail - (uintptr_t)&ctl->head) >= UCS_SYS_CACHE_LINE_SIZE);

    iface->recv_fifo_ctl = ctl;

    ucs_assert(iface->shared_mem != NULL);
    return UCS_OK;
}

static ucs_status_t uct_mm_iface_create_signal_fd(uct_mm_iface_t *iface)
{
    ucs_status_t status;
    socklen_t addrlen;
    struct sockaddr_un bind_addr;
    int ret;

    /* Create a UNIX domain socket to send and receive wakeup signal from remote processes */
    iface->signal_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (iface->signal_fd < 0) {
        ucs_error("Failed to create unix domain socket for signal: %m");
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    /* Set the signal socket to non-blocking mode */
    status = ucs_sys_fcntl_modfl(iface->signal_fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_close;
    }

    /* Bind the signal socket to automatic address */
    bind_addr.sun_family = AF_UNIX;
    memset(bind_addr.sun_path, 0, sizeof(bind_addr.sun_path));
    ret = bind(iface->signal_fd, (struct sockaddr*)&bind_addr, sizeof(sa_family_t));
    if (ret < 0) {
        ucs_error("Failed to auto-bind unix domain socket: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_close;
    }

    /* Share the socket address on the FIFO control area, so we would not have
     * to enlarge the interface address size.
     */
    addrlen = sizeof(struct sockaddr_un);
    memset(&iface->recv_fifo_ctl->signal_sockaddr, 0, addrlen);
    ret = getsockname(iface->signal_fd,
                      (struct sockaddr *)&iface->recv_fifo_ctl->signal_sockaddr,
                      &addrlen);
    if (ret < 0) {
        ucs_error("Failed to retrieve unix domain socket address: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_close;
    }

    iface->recv_fifo_ctl->signal_addrlen = addrlen;
    return UCS_OK;

err_close:
    close(iface->signal_fd);
err:
    return status;
}

static UCS_CLASS_INIT_FUNC(uct_mm_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_mm_iface_config_t *mm_config = ucs_derived_of(tl_config, uct_mm_iface_config_t);
    uct_mm_fifo_element_t* fifo_elem_p;
    ucs_status_t status;
    unsigned i;

    ucs_assert(params->open_mode & UCT_IFACE_OPEN_MODE_DEVICE);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_mm_iface_ops, md, worker,
                              params, tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(UCT_MM_TL_NAME));

    ucs_trace_func("Creating an MM iface=%p worker=%p", self, worker);

    /* check that the fifo size, from the user, is a power of two and bigger than 1 */
    if ((mm_config->fifo_size <= 1) || ucs_is_pow2(mm_config->fifo_size) != 1) {
        ucs_error("The MM FIFO size must be a power of two and bigger than 1.");
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    /* check the value defining the FIFO batch release */
    if ((mm_config->release_fifo_factor < 0) || (mm_config->release_fifo_factor >= 1)) {
        ucs_error("The MM release FIFO factor must be: (0 =< factor < 1).");
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    /* check the value defining the size of the FIFO element */
    if (mm_config->super.max_short <= sizeof(uct_mm_fifo_element_t)) {
        ucs_error("The UCT_MM_MAX_SHORT parameter must be larger than the FIFO "
                  "element header size. ( > %ld bytes).",
                  sizeof(uct_mm_fifo_element_t));
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    self->config.fifo_size         = mm_config->fifo_size;
    self->config.fifo_elem_size    = mm_config->super.max_short;
    self->config.seg_size          = mm_config->super.max_bcopy;
    self->fifo_release_factor_mask = UCS_MASK(ucs_ilog2(ucs_max((int)
                                     (mm_config->fifo_size * mm_config->release_fifo_factor),
                                     1)));
    self->fifo_mask                = mm_config->fifo_size - 1;
    self->fifo_shift               = ucs_count_zero_bits(mm_config->fifo_size);
    self->rx_headroom              = params->rx_headroom;
    self->release_desc.cb          = uct_mm_iface_release_desc;

    /* create the receive FIFO */
    /* use specific allocator to allocate and attach memory and check the
     * requested hugetlb allocation mode */
    status = uct_mm_allocate_fifo_mem(self, mm_config, md);
    if (status != UCS_OK) {
        goto err;
    }

    self->recv_fifo_ctl->head   = 0;
    self->recv_fifo_ctl->tail   = 0;
    self->read_index            = 0;

    status = uct_mm_iface_create_signal_fd(self);
    if (status != UCS_OK) {
        goto err_free_fifo;
    }

    /* create a memory pool for receive descriptors */
    status = uct_iface_mpool_init(&self->super,
                                  &self->recv_desc_mp,
                                  sizeof(uct_mm_recv_desc_t) + params->rx_headroom +
                                  self->config.seg_size,
                                  sizeof(uct_mm_recv_desc_t),
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &mm_config->mp,
                                  512,
                                  uct_mm_iface_recv_desc_init,
                                  "mm_recv_desc");
    if (status != UCS_OK) {
        ucs_error("Failed to create a receive descriptor memory pool for the MM transport");
        goto err_close_signal_fd;
    }

    ucs_mpool_grow(&self->recv_desc_mp, mm_config->fifo_size * 2);

    /* set the first receive descriptor */
    self->last_recv_desc = ucs_mpool_get(&self->recv_desc_mp);
    VALGRIND_MAKE_MEM_DEFINED(self->last_recv_desc, sizeof(*(self->last_recv_desc)));
    if (self->last_recv_desc == NULL) {
        ucs_error("Failed to get the first receive descriptor");
        status = UCS_ERR_NO_RESOURCE;
        goto destroy_recv_mpool;
    }

    /* initiate the owner bit in all the FIFO elements and assign a receive descriptor
     * per every FIFO element */
    for (i = 0; i < mm_config->fifo_size; i++) {
        fifo_elem_p = UCT_MM_IFACE_GET_FIFO_ELEM(self, self->recv_fifo_elements, i);
        fifo_elem_p->flags = UCT_MM_FIFO_ELEM_FLAG_OWNER;

        status = uct_mm_assign_desc_to_fifo_elem(self, fifo_elem_p, 1);
        if (status != UCS_OK) {
            ucs_error("Failed to allocate a descriptor for MM");
            goto destroy_descs;
        }
    }

    ucs_arbiter_init(&self->arbiter);

    ucs_debug("Created an MM iface. FIFO mm id: %zu", self->fifo_mm_id);
    return UCS_OK;

destroy_descs:
    uct_mm_iface_free_rx_descs(self, i);
    ucs_mpool_put(self->last_recv_desc);
destroy_recv_mpool:
    ucs_mpool_cleanup(&self->recv_desc_mp, 1);
err_close_signal_fd:
    close(self->signal_fd);
err_free_fifo:
    uct_mm_md_mapper_ops(md)->free(self->shared_mem, self->fifo_mm_id,
                                   UCT_MM_GET_FIFO_SIZE(self), self->path);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_mm_iface_t)
{
    ucs_status_t status;
    size_t size_to_free;

    uct_base_iface_progress_disable(&self->super.super,
                                   UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    /* return all the descriptors that are now 'assigned' to the FIFO,
     * to their mpool */
    uct_mm_iface_free_rx_descs(self, self->config.fifo_size);

    ucs_mpool_put(self->last_recv_desc);
    ucs_mpool_cleanup(&self->recv_desc_mp, 1);
    close(self->signal_fd);

    size_to_free = UCT_MM_GET_FIFO_SIZE(self);

    /* release the memory allocated for the FIFO */
    status = uct_mm_md_mapper_ops(self->super.md)->free(self->shared_mem,
                                                        self->fifo_mm_id,
                                                        size_to_free, self->path);
    if (status != UCS_OK) {
        ucs_warn("Unable to release shared memory segment: %m");
    }

    ucs_arbiter_cleanup(&self->arbiter);
}

UCS_CLASS_DEFINE(uct_mm_iface_t, uct_base_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_mm_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t *,
                                 const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_mm_iface_t, uct_iface_t);

static ucs_status_t uct_mm_query_tl_resources(uct_md_h md,
                                              uct_tl_resource_desc_t **resource_p,
                                              unsigned *num_resources_p)
{
    uct_tl_resource_desc_t *resource;

    resource = ucs_calloc(1, sizeof(uct_tl_resource_desc_t), "resource desc");
    if (NULL == resource) {
        ucs_error("Failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }

    ucs_snprintf_zero(resource->tl_name, sizeof(resource->tl_name), "%s",
                      UCT_MM_TL_NAME);
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s",
                      md->component->name);
    resource->dev_type = UCT_DEVICE_TYPE_SHM;

    *num_resources_p = 1;
    *resource_p      = resource;
    return UCS_OK;
}

UCT_TL_COMPONENT_DEFINE(uct_mm_tl,
                        uct_mm_query_tl_resources,
                        uct_mm_iface_t,
                        UCT_MM_TL_NAME,
                        "MM_",
                        uct_mm_iface_config_table,
                        uct_mm_iface_config_t);
