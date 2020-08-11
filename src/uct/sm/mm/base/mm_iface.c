/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "mm_iface.h"
#include "mm_ep.h"

#include <uct/base/uct_worker.h>
#include <uct/sm/base/sm_ep.h>
#include <ucs/arch/atomic.h>
#include <ucs/arch/bitops.h>
#include <ucs/async/async.h>
#include <ucs/sys/string.h>
#include <sys/poll.h>


/* Maximal number of events to clear from the signaling pipe in single call */
#define UCT_MM_IFACE_MAX_SIG_EVENTS  32


ucs_config_field_t uct_mm_iface_config_table[] = {
    {"SM_", "ALLOC=md,mmap,heap", NULL,
     ucs_offsetof(uct_mm_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_sm_iface_config_table)},

    {"FIFO_SIZE", "64",
     "Size of the receive FIFO in the memory-map UCTs.",
     ucs_offsetof(uct_mm_iface_config_t, fifo_size), UCS_CONFIG_TYPE_UINT},

    {"SEG_SIZE", "8256",
     "Size of send/receive buffers for copy-out sends.",
     ucs_offsetof(uct_mm_iface_config_t, seg_size), UCS_CONFIG_TYPE_MEMUNITS},

    {"FIFO_RELEASE_FACTOR", "0.5",
     "Frequency of resource releasing on the receiver's side in the MM UCT.\n"
     "This value refers to the percentage of the FIFO size. (must be >= 0 and < 1).",
     ucs_offsetof(uct_mm_iface_config_t, release_fifo_factor), UCS_CONFIG_TYPE_DOUBLE},

    UCT_IFACE_MPOOL_CONFIG_FIELDS("RX_", -1, 512, "receive",
                                  ucs_offsetof(uct_mm_iface_config_t, mp), ""),

    {"FIFO_HUGETLB", "no",
     "Enable using huge pages for internal shared memory buffers."
     "Possible values are:\n"
     " y   - Allocate memory using huge pages only.\n"
     " n   - Allocate memory using regular pages only.\n"
     " try - Try to allocate memory using huge pages and if it fails, allocate regular pages.",
     ucs_offsetof(uct_mm_iface_config_t, hugetlb_mode), UCS_CONFIG_TYPE_TERNARY},

    {"FIFO_ELEM_SIZE", "128",
     "Size of the FIFO element size (data + header) in the MM UCTs.",
     ucs_offsetof(uct_mm_iface_config_t, fifo_elem_size), UCS_CONFIG_TYPE_UINT},

    {"FIFO_MAX_POLL", UCS_PP_MAKE_STRING(UCT_MM_IFACE_FIFO_MAX_POLL),
     "Maximal number of receive completions to pick during RX poll",
     ucs_offsetof(uct_mm_iface_config_t, fifo_max_poll), UCS_CONFIG_TYPE_ULUNITS},

    {NULL}
};

static ucs_status_t uct_mm_iface_get_address(uct_iface_t *tl_iface,
                                             uct_iface_addr_t *addr)
{
    uct_mm_iface_t      *iface      = ucs_derived_of(tl_iface, uct_mm_iface_t);
    uct_mm_md_t         *md         = ucs_derived_of(iface->super.super.md,
                                                     uct_mm_md_t);
    uct_mm_iface_addr_t *iface_addr = (void*)addr;
    uct_mm_seg_t        *seg        = iface->recv_fifo_mem.memh;

    iface_addr->fifo_seg_id = seg->seg_id;
    return uct_mm_md_mapper_ops(md)->iface_addr_pack(md, iface_addr + 1);
}

static int
uct_mm_iface_is_reachable(const uct_iface_h tl_iface,
                          const uct_device_addr_t *dev_addr,
                          const uct_iface_addr_t *tl_iface_addr)
{
    uct_mm_iface_t      *iface      = ucs_derived_of(tl_iface, uct_mm_iface_t);
    uct_mm_md_t         *md         = ucs_derived_of(iface->super.super.md,
                                                     uct_mm_md_t);
    uct_mm_iface_addr_t *iface_addr = (void*)tl_iface_addr;

    if (!uct_sm_iface_is_reachable(tl_iface, dev_addr, tl_iface_addr)) {
        return 0;
    }

    return uct_mm_md_mapper_ops(md)->is_reachable(md, iface_addr->fifo_seg_id,
                                                  iface_addr + 1);
}

void uct_mm_iface_release_desc(uct_recv_desc_t *self, void *desc)
{
    void *mm_desc;

    mm_desc = UCS_PTR_BYTE_OFFSET(desc, -sizeof(uct_mm_recv_desc_t));
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
    uct_mm_md_t    *md    = ucs_derived_of(iface->super.super.md, uct_mm_md_t);

    uct_base_iface_query(&iface->super.super, iface_attr);

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

    iface_attr->iface_addr_len          = sizeof(uct_mm_iface_addr_t) +
                                          md->iface_addr_len;
    iface_attr->device_addr_len         = uct_sm_iface_get_device_addr_len();
    iface_attr->ep_addr_len             = 0;
    iface_attr->max_conn_priv           = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_PUT_SHORT           |
                                          UCT_IFACE_FLAG_PUT_BCOPY           |
                                          UCT_IFACE_FLAG_ATOMIC_CPU          |
                                          UCT_IFACE_FLAG_GET_BCOPY           |
                                          UCT_IFACE_FLAG_AM_SHORT            |
                                          UCT_IFACE_FLAG_AM_BCOPY            |
                                          UCT_IFACE_FLAG_PENDING             |
                                          UCT_IFACE_FLAG_CB_SYNC             |
                                          UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    iface_attr->cap.event_flags         = UCT_IFACE_FLAG_EVENT_SEND_COMP     |
                                          UCT_IFACE_FLAG_EVENT_RECV_SIG      |
                                          UCT_IFACE_FLAG_EVENT_FD;

    iface_attr->cap.atomic32.op_flags   =
    iface_attr->cap.atomic64.op_flags   = UCS_BIT(UCT_ATOMIC_OP_ADD)         |
                                          UCS_BIT(UCT_ATOMIC_OP_AND)         |
                                          UCS_BIT(UCT_ATOMIC_OP_OR)          |
                                          UCS_BIT(UCT_ATOMIC_OP_XOR);
    iface_attr->cap.atomic32.fop_flags  =
    iface_attr->cap.atomic64.fop_flags  = UCS_BIT(UCT_ATOMIC_OP_ADD)         |
                                          UCS_BIT(UCT_ATOMIC_OP_AND)         |
                                          UCS_BIT(UCT_ATOMIC_OP_OR)          |
                                          UCS_BIT(UCT_ATOMIC_OP_XOR)         |
                                          UCS_BIT(UCT_ATOMIC_OP_SWAP)        |
                                          UCS_BIT(UCT_ATOMIC_OP_CSWAP);

    iface_attr->latency                 = ucs_linear_func_make(80e-9, 0); /* 80 ns */
    iface_attr->bandwidth.dedicated     = iface->super.config.bandwidth;
    iface_attr->bandwidth.shared        = 0;
    iface_attr->overhead                = 10e-9; /* 10 ns */
    iface_attr->priority                = 0;

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_mm_progress_fifo_tail(uct_mm_iface_t *iface)
{
    /* don't progress the tail every time - release in batches. improves performance */
    if (iface->read_index & iface->fifo_release_factor_mask) {
        return;
    }

    iface->recv_fifo_ctl->tail = iface->read_index;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_mm_assign_desc_to_fifo_elem(uct_mm_iface_t *iface,
                                uct_mm_fifo_element_t *elem,
                                unsigned need_new_desc)
{
    uct_mm_recv_desc_t *desc;

    if (!need_new_desc) {
        desc = iface->last_recv_desc;
    } else {
        UCT_TL_IFACE_GET_RX_DESC(&iface->super.super, &iface->recv_desc_mp, desc,
                                 return UCS_ERR_NO_RESOURCE);
    }

    elem->desc      = desc->info;
    elem->desc_data = UCS_PTR_BYTE_OFFSET(desc + 1, iface->rx_headroom);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_mm_iface_process_recv(uct_mm_iface_t *iface,
                          uct_mm_fifo_element_t* elem)
{
    ucs_status_t status;
    void         *data;

    if (ucs_likely(elem->flags & UCT_MM_FIFO_ELEM_FLAG_INLINE)) {
        /* read short (inline) messages from the FIFO elements */
        uct_iface_trace_am(&iface->super.super, UCT_AM_TRACE_TYPE_RECV,
                           elem->am_id, elem + 1, elem->length, "RX: AM_SHORT");
        uct_mm_iface_invoke_am(iface, elem->am_id, elem + 1, elem->length, 0);
        return;
    }

    /* check the memory pool to make sure that there is a new descriptor available */
    if (ucs_unlikely(iface->last_recv_desc == NULL)) {
        UCT_TL_IFACE_GET_RX_DESC(&iface->super.super, &iface->recv_desc_mp,
                                 iface->last_recv_desc, return);
    }

    /* read bcopy messages from the receive descriptors */
    data = elem->desc_data;
    VALGRIND_MAKE_MEM_DEFINED(data, elem->length);

    uct_iface_trace_am(&iface->super.super, UCT_AM_TRACE_TYPE_RECV,
                       elem->am_id, data, elem->length, "RX: AM_BCOPY");

    status = uct_mm_iface_invoke_am(iface, elem->am_id, data, elem->length,
                                    UCT_CB_PARAM_FLAG_DESC);
    if (status != UCS_OK) {
        /* assign a new receive descriptor to this FIFO element.*/
        uct_mm_assign_desc_to_fifo_elem(iface, elem, 0);
        /* the last_recv_desc is in use. get a new descriptor for it */
        UCT_TL_IFACE_GET_RX_DESC(&iface->super.super, &iface->recv_desc_mp,
                                 iface->last_recv_desc, ucs_debug("recv mpool is empty"));
    }
}

static UCS_F_ALWAYS_INLINE int
uct_mm_iface_fifo_has_new_data(uct_mm_iface_t *iface)
{
    /* check the read_index to see if there is a new item to read
     * (checking the owner bit) */
    return (((iface->read_index >> iface->fifo_shift) & 1) ==
            (iface->read_index_elem->flags & 1));
}

static UCS_F_ALWAYS_INLINE unsigned
uct_mm_iface_poll_fifo(uct_mm_iface_t *iface)
{
    if (!uct_mm_iface_fifo_has_new_data(iface)) {
        return 0;
    }

    /* read from read_index_elem */
    ucs_memory_cpu_load_fence();
    ucs_assert(iface->read_index <= iface->recv_fifo_ctl->head);

    uct_mm_iface_process_recv(iface, iface->read_index_elem);

    /* raise the read_index */
    iface->read_index++;

    /* the next fifo_element which the read_index points to */
    iface->read_index_elem =
        UCT_MM_IFACE_GET_FIFO_ELEM(iface, iface->recv_fifo_elems,
                                   (iface->read_index & iface->fifo_mask));

    uct_mm_progress_fifo_tail(iface);

    return 1;
}

static UCS_F_ALWAYS_INLINE void
uct_mm_iface_fifo_window_adjust(uct_mm_iface_t *iface,
                                unsigned fifo_poll_count)
{
    if (fifo_poll_count < iface->fifo_poll_count) {
        iface->fifo_poll_count = ucs_max(iface->fifo_poll_count /
                                         UCT_MM_IFACE_FIFO_MD_FACTOR,
                                         UCT_MM_IFACE_FIFO_MIN_POLL);
        iface->fifo_prev_wnd_cons = 0;
        return;
    }

    ucs_assert(fifo_poll_count == iface->fifo_poll_count);

    if (iface->fifo_prev_wnd_cons) {
        /* Increase FIFO window size if it was fully consumed
         * during the previous iface progress call in order
         * to prevent the situation when the window will be
         * adjusted to [MIN, MIN + 1, MIN, MIN + 1, ...] that
         * is harmful to latency */
        iface->fifo_poll_count = ucs_min(iface->fifo_poll_count +
                                         UCT_MM_IFACE_FIFO_AI_VALUE,
                                         iface->config.fifo_max_poll);
    } else {
        iface->fifo_prev_wnd_cons = 1;
    }
}

static unsigned uct_mm_iface_progress(uct_iface_h tl_iface)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_iface, uct_mm_iface_t);
    unsigned total_count  = 0;
    unsigned count;

    ucs_assert(iface->fifo_poll_count >= UCT_MM_IFACE_FIFO_MIN_POLL);

    /* progress receive */
    do {
        count = uct_mm_iface_poll_fifo(iface);
        ucs_assert(count < 2);
        total_count += count;
        ucs_assert(total_count < UINT_MAX);
    } while ((count != 0) && (total_count < iface->fifo_poll_count));

    uct_mm_iface_fifo_window_adjust(iface, total_count);

    /* progress the pending sends (if there are any) */
    ucs_arbiter_dispatch(&iface->arbiter, 1, uct_mm_ep_process_pending,
                         &total_count);

    return total_count;
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
    .ep_atomic_cswap64        = uct_sm_ep_atomic_cswap64,
    .ep_atomic64_post         = uct_sm_ep_atomic64_post,
    .ep_atomic64_fetch        = uct_sm_ep_atomic64_fetch,
    .ep_atomic_cswap32        = uct_sm_ep_atomic_cswap32,
    .ep_atomic32_post         = uct_sm_ep_atomic32_post,
    .ep_atomic32_fetch        = uct_sm_ep_atomic32_fetch,
    .ep_pending_add           = uct_mm_ep_pending_add,
    .ep_pending_purge         = uct_mm_ep_pending_purge,
    .ep_flush                 = uct_mm_ep_flush,
    .ep_fence                 = uct_sm_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_mm_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_mm_ep_t),
    .iface_flush              = uct_mm_iface_flush,
    .iface_fence              = uct_sm_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_mm_iface_progress,
    .iface_event_fd_get       = uct_mm_iface_event_fd_get,
    .iface_event_arm          = uct_mm_iface_event_fd_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_mm_iface_t),
    .iface_query              = uct_mm_iface_query,
    .iface_get_device_address = uct_sm_iface_get_device_address,
    .iface_get_address        = uct_mm_iface_get_address,
    .iface_is_reachable       = uct_mm_iface_is_reachable
};

static void uct_mm_iface_recv_desc_init(uct_iface_h tl_iface, void *obj,
                                        uct_mem_h memh)
{
    uct_mm_iface_t     *iface = ucs_derived_of(tl_iface, uct_mm_iface_t);
    uct_mm_recv_desc_t *desc  = obj;
    uct_mm_seg_t        *seg  = memh;
    size_t offset;

    if (seg->length > UINT_MAX) {
        ucs_error("mm: shared memory segment length cannot exceed %u", UINT_MAX);
        desc->info.seg_id   = UINT64_MAX;
        desc->info.seg_size = 0;
        desc->info.offset   = 0;
        return;
    }

    offset = UCS_PTR_BYTE_DIFF(seg->address, desc + 1) + iface->rx_headroom;
    ucs_assert(offset <= UINT_MAX);

    desc->info.seg_id   = seg->seg_id;
    desc->info.seg_size = seg->length;
    desc->info.offset   = offset;
}

static void uct_mm_iface_free_rx_descs(uct_mm_iface_t *iface, unsigned num_elems)
{
    uct_mm_fifo_element_t *elem;
    uct_mm_recv_desc_t *desc;
    unsigned i;

    for (i = 0; i < num_elems; i++) {
        elem = UCT_MM_IFACE_GET_FIFO_ELEM(iface, iface->recv_fifo_elems, i);
        desc = (uct_mm_recv_desc_t*)UCS_PTR_BYTE_OFFSET(elem->desc_data,
                                                        -iface->rx_headroom) - 1;
        ucs_mpool_put(desc);
    }
}

void uct_mm_iface_set_fifo_ptrs(void *fifo_mem, uct_mm_fifo_ctl_t **fifo_ctl_p,
                                void **fifo_elems_p)
{
    uct_mm_fifo_ctl_t *fifo_ctl;

    /* initiate the the uct_mm_fifo_ctl struct, holding the head and the tail */
    fifo_ctl = (uct_mm_fifo_ctl_t*)ucs_align_up_pow2
                    ((uintptr_t)fifo_mem, UCS_SYS_CACHE_LINE_SIZE);

    /* Make sure head and tail are cache-aligned, and not on same cacheline, to
     * avoid false-sharing.
     */
    ucs_assert_always(
        (((uintptr_t)&fifo_ctl->head) % UCS_SYS_CACHE_LINE_SIZE) == 0);
    ucs_assert_always(
        (((uintptr_t)&fifo_ctl->tail) % UCS_SYS_CACHE_LINE_SIZE) == 0);
    ucs_assert_always(
        ((uintptr_t)&fifo_ctl->tail - (uintptr_t)&fifo_ctl->head) >= UCS_SYS_CACHE_LINE_SIZE);

    /* initiate the pointer to the beginning of the first FIFO element */
    *fifo_ctl_p   = fifo_ctl;
    *fifo_elems_p = UCS_PTR_BYTE_OFFSET(fifo_ctl, UCT_MM_FIFO_CTL_SIZE);
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
                      (struct sockaddr *)ucs_unaligned_ptr(&iface->recv_fifo_ctl->signal_sockaddr),
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

static void uct_mm_iface_log_created(uct_mm_iface_t *iface)
{
    uct_mm_seg_t UCS_V_UNUSED *seg = iface->recv_fifo_mem.memh;

    ucs_debug("created mm iface %p FIFO id 0x%"PRIx64
              " va %p size %zu (%u x %u elems)",
              iface, seg->seg_id, seg->address, seg->length,
              iface->config.fifo_elem_size, iface->config.fifo_size);
}

static UCS_CLASS_INIT_FUNC(uct_mm_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_mm_iface_config_t *mm_config =
                    ucs_derived_of(tl_config, uct_mm_iface_config_t);
    uct_mm_fifo_element_t* fifo_elem_p;
    ucs_status_t status;
    unsigned i;

    UCS_CLASS_CALL_SUPER_INIT(uct_sm_iface_t, &uct_mm_iface_ops, md,
                              worker, params, tl_config);

    if (ucs_derived_of(worker, uct_priv_worker_t)->thread_mode == UCS_THREAD_MODE_MULTI) {
        ucs_error("Shared memory transport does not support multi-threaded worker");
        return UCS_ERR_INVALID_PARAM;
    }

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
    if (mm_config->fifo_elem_size <= sizeof(uct_mm_fifo_element_t)) {
        ucs_error("The UCX_MM_FIFO_ELEM_SIZE parameter (%u) must be larger "
                  "than the FIFO element header size (%ld bytes).",
                  mm_config->fifo_elem_size, sizeof(uct_mm_fifo_element_t));
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    self->config.fifo_size         = mm_config->fifo_size;
    self->config.fifo_elem_size    = mm_config->fifo_elem_size;
    self->config.seg_size          = mm_config->seg_size;
    self->config.fifo_max_poll     = ((mm_config->fifo_max_poll == UCS_ULUNITS_AUTO) ?
                                      UCT_MM_IFACE_FIFO_MAX_POLL :
                                      /* trim by the maximum unsigned integer value */
                                      ucs_min(mm_config->fifo_max_poll, UINT_MAX));
    self->fifo_prev_wnd_cons       = 0;
    self->fifo_poll_count          = self->config.fifo_max_poll;
    /* cppcheck-suppress internalAstError */
    self->fifo_release_factor_mask = UCS_MASK(ucs_ilog2(ucs_max((int)
                                     (mm_config->fifo_size * mm_config->release_fifo_factor),
                                     1)));
    self->fifo_mask                = self->config.fifo_size - 1;
    self->fifo_shift               = ucs_count_trailing_zero_bits(mm_config->fifo_size);
    self->rx_headroom              = (params->field_mask &
                                      UCT_IFACE_PARAM_FIELD_RX_HEADROOM) ?
                                     params->rx_headroom : 0;
    self->release_desc.cb          = uct_mm_iface_release_desc;

    /* Allocate the receive FIFO */
    status = uct_iface_mem_alloc(&self->super.super.super,
                                 UCT_MM_GET_FIFO_SIZE(self),
                                 UCT_MD_MEM_ACCESS_ALL, "mm_recv_fifo",
                                 &self->recv_fifo_mem);
    if (status != UCS_OK) {
        ucs_error("mm_iface failed to allocate receive FIFO");
        return status;
    }

    uct_mm_iface_set_fifo_ptrs(self->recv_fifo_mem.address,
                               &self->recv_fifo_ctl, &self->recv_fifo_elems);
    self->recv_fifo_ctl->head = 0;
    self->recv_fifo_ctl->tail = 0;
    self->read_index          = 0;
    self->read_index_elem     = UCT_MM_IFACE_GET_FIFO_ELEM(self,
                                                           self->recv_fifo_elems,
                                                           self->read_index);

    /* create a unix file descriptor to receive event notifications */
    status = uct_mm_iface_create_signal_fd(self);
    if (status != UCS_OK) {
        goto err_free_fifo;
    }

    /* create a memory pool for receive descriptors */
    status = uct_iface_mpool_init(&self->super.super,
                                  &self->recv_desc_mp,
                                  sizeof(uct_mm_recv_desc_t) + self->rx_headroom +
                                  self->config.seg_size,
                                  sizeof(uct_mm_recv_desc_t),
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &mm_config->mp,
                                  mm_config->mp.bufs_grow,
                                  uct_mm_iface_recv_desc_init,
                                  "mm_recv_desc");
    if (status != UCS_OK) {
        ucs_error("failed to create a receive descriptor memory pool for the MM transport");
        goto err_close_signal_fd;
    }

    /* set the first receive descriptor */
    self->last_recv_desc = ucs_mpool_get(&self->recv_desc_mp);
    VALGRIND_MAKE_MEM_DEFINED(self->last_recv_desc, sizeof(*(self->last_recv_desc)));
    if (self->last_recv_desc == NULL) {
        ucs_error("failed to get the first receive descriptor");
        status = UCS_ERR_NO_RESOURCE;
        goto destroy_recv_mpool;
    }

    /* initiate the owner bit in all the FIFO elements and assign a receive descriptor
     * per every FIFO element */
    for (i = 0; i < mm_config->fifo_size; i++) {
        fifo_elem_p = UCT_MM_IFACE_GET_FIFO_ELEM(self, self->recv_fifo_elems, i);
        fifo_elem_p->flags = UCT_MM_FIFO_ELEM_FLAG_OWNER;

        status = uct_mm_assign_desc_to_fifo_elem(self, fifo_elem_p, 1);
        if (status != UCS_OK) {
            ucs_error("failed to allocate a descriptor for MM");
            goto destroy_descs;
        }
    }

    ucs_arbiter_init(&self->arbiter);
    uct_mm_iface_log_created(self);

    return UCS_OK;

destroy_descs:
    uct_mm_iface_free_rx_descs(self, i);
    ucs_mpool_put(self->last_recv_desc);
destroy_recv_mpool:
    ucs_mpool_cleanup(&self->recv_desc_mp, 1);
err_close_signal_fd:
    close(self->signal_fd);
err_free_fifo:
    uct_iface_mem_free(&self->recv_fifo_mem);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_mm_iface_t)
{
    uct_base_iface_progress_disable(&self->super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    /* return all the descriptors that are now 'assigned' to the FIFO,
     * to their mpool */
    uct_mm_iface_free_rx_descs(self, self->config.fifo_size);

    ucs_mpool_put(self->last_recv_desc);
    ucs_mpool_cleanup(&self->recv_desc_mp, 1);
    close(self->signal_fd);
    uct_iface_mem_free(&self->recv_fifo_mem);
    ucs_arbiter_cleanup(&self->arbiter);
}

UCS_CLASS_DEFINE(uct_mm_iface_t, uct_base_iface_t);

UCS_CLASS_DEFINE_NEW_FUNC(uct_mm_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_mm_iface_t, uct_iface_t);
