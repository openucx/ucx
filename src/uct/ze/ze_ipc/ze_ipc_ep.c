/*
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ze_ipc_ep.h"
#include "ze_ipc_iface.h"
#include "ze_ipc_md.h"
#include "ze_ipc_cache.h"
#include <uct/ze/base/ze_base.h>

#include <uct/base/uct_log.h>
#include <uct/base/uct_iov.inl>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/math.h>
#include <ucs/type/class.h>
#include <ucs/profile/profile.h>

#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>

#include <stdio.h>
#include <time.h>

#define UCT_ZE_IPC_PUT 0
#define UCT_ZE_IPC_GET 1

/*
 *
 * @param remote_pid  PID of the process that owns the fd
 * @param remote_fd   File descriptor in the remote process
 * @return            Local fd on success, -1 on error
 */
int uct_ze_ipc_dup_fd_from_pid(pid_t remote_pid, int remote_fd)
{
#if defined(SYS_pidfd_open) && defined(SYS_pidfd_getfd)
    int pidfd, local_fd;

    /* Skip if same process */
    if (remote_pid == getpid()) {
        return remote_fd;
    }

    /* Open pidfd for the remote process */
    pidfd = syscall(SYS_pidfd_open, remote_pid, 0);
    if (pidfd < 0) {
        ucs_error("ze_ipc: pidfd_open(%d) failed: %m", remote_pid);
        return -1;
    }

    /* Duplicate the fd from remote process */
    local_fd = syscall(SYS_pidfd_getfd, pidfd, remote_fd, 0);
    if (local_fd < 0) {
        ucs_error("ze_ipc: pidfd_getfd(pidfd=%d, remote_fd=%d) failed: %m",
                  pidfd, remote_fd);
        close(pidfd);
        return -1;
    }

    close(pidfd);
    ucs_debug("ze_ipc: duplicated fd %d from pid %d -> local fd %d (uncached)",
              remote_fd, remote_pid, local_fd);
    return local_fd;
#else
    ucs_error("ze_ipc: pidfd_getfd not supported on this system");
    return -1;
#endif
}


static UCS_CLASS_INIT_FUNC(uct_ze_ipc_ep_t, const uct_ep_params_t *params)
{
    uct_ze_ipc_iface_t *iface = ucs_derived_of(params->iface,
                                               uct_ze_ipc_iface_t);

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    self->remote_pid = *(const pid_t*)params->iface_addr;

    ucs_info("ze_ipc_ep: created endpoint to remote pid %d (local pid %d)",
             self->remote_pid, getpid());
    return UCS_OK;
}


static UCS_CLASS_CLEANUP_FUNC(uct_ze_ipc_ep_t)
{
}


UCS_CLASS_DEFINE(uct_ze_ipc_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_ze_ipc_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_ze_ipc_ep_t, uct_ep_t);


#define uct_ze_ipc_trace_data(_addr, _rkey, _fmt, ...)     \
    ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_addr), (_rkey))


int uct_ze_ipc_ep_is_connected(const uct_ep_h tl_ep,
                               const uct_ep_is_connected_params_t *params)
{
    const uct_ze_ipc_ep_t *ep = ucs_derived_of(tl_ep, uct_ze_ipc_ep_t);

    if (!uct_base_ep_is_connected(tl_ep, params)) {
        return 0;
    }

    return ep->remote_pid == *(pid_t*)params->iface_addr;
}


static UCS_F_ALWAYS_INLINE ucs_status_t
uct_ze_ipc_post_copy(uct_ep_h tl_ep, uint64_t remote_addr,
                     const uct_iov_t *iov, uct_rkey_t rkey,
                     uct_completion_t *comp, int direction)
{
    uct_ze_ipc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ze_ipc_iface_t);
    uct_ze_ipc_ep_t *ep       = ucs_derived_of(tl_ep, uct_ze_ipc_ep_t);
    uct_ze_ipc_key_t *key     = (uct_ze_ipc_key_t *)rkey;
    uct_ze_ipc_event_desc_t *event_desc;
    uct_ze_ipc_queue_desc_t *q_desc;
    ze_event_handle_t event = NULL;
    void *mapped_addr = NULL;
    void *mapped_rem_addr;
    void *dst, *src;
    size_t offset;
    ze_result_t ret;
    ucs_status_t status;
    int local_fd;
    int event_index = -1;
    unsigned cmd_list_idx;
    struct timespec start1, start2, start3, start4, end;
    double elapsed_ms1, elapsed_ms2, elapsed_ms3, elapsed_ms4, elapsed_ms5;

    clock_gettime(CLOCK_MONOTONIC, &start1);

    if (ucs_unlikely(iov[0].length == 0)) {
        ucs_trace_data("Zero length request: skip it");
        return UCS_OK;
    }

    /* Use cache to map IPC handle (with pidfd caching) */
    status = uct_ze_ipc_map_memhandle(iface, key, iface->ze_context, iface->ze_device,
                                      &mapped_addr, &local_fd);
    if (status != UCS_OK) {
        ucs_error("ze_ipc_ep: uct_ze_ipc_map_memhandle failed");
        return status;
    }
    clock_gettime(CLOCK_MONOTONIC, &start2);

    ucs_debug("ze_ipc_ep: IPC handle mapped (cached), mapped_addr=%p", mapped_addr);

    /* Calculate offset within the allocation */
    offset          = remote_addr - key->address;
    mapped_rem_addr = (void *)((uintptr_t)mapped_addr + offset);

    /* Allocate event from shared pool for performance */
    event_index = uct_ze_ipc_alloc_event(iface, &event);
    if (event_index < 0) {
        ucs_error("failed to allocate event from shared pool");
        uct_ze_ipc_unmap_memhandle(ep->remote_pid, key->address, mapped_addr,
                                   iface->ze_context, local_fd,
                                   iface->config.enable_cache);
        return UCS_ERR_NO_RESOURCE;
    }

    /* Allocate event descriptor */
    event_desc = ucs_malloc(sizeof(*event_desc), "uct_ze_ipc_event_desc_t");
    if (event_desc == NULL) {
        ucs_error("failed to allocate event descriptor");
        uct_ze_ipc_free_event(iface, event, event_index);
        uct_ze_ipc_unmap_memhandle(ep->remote_pid, key->address, mapped_addr,
                                   iface->ze_context, local_fd,
                                   iface->config.enable_cache);
        return UCS_ERR_NO_MEMORY;
    }

    /* Store information for cache-based cleanup and event tracking */
    event_desc->event       = event;
    event_desc->event_pool  = NULL;  /* Using shared pool, not private */
    event_desc->event_index = event_index;
    event_desc->dup_fd      = local_fd;
    event_desc->pid         = ep->remote_pid;
    event_desc->address     = key->address;

    /* Set up source and destination based on direction */
    if (direction == UCT_ZE_IPC_PUT) {
        dst = mapped_rem_addr;
        src = iov[0].buffer;
    } else {
        dst = iov[0].buffer;
        src = mapped_rem_addr;
    }

    clock_gettime(CLOCK_MONOTONIC, &start3);

    /*
     * Select command list using round-robin scheduling
     * Similar to CUDA IPC's stream selection: key->stream_id % iface->config.max_streams
     * This distributes copy operations across multiple command lists for parallel execution
     */
    cmd_list_idx = iface->next_cmd_list % iface->num_cmd_lists;
    iface->next_cmd_list++;

    q_desc = &iface->queue_desc[cmd_list_idx];

    /*
     * Append memory copy to selected immediate command list with event signaling.
     * Immediate command lists execute asynchronously without needing
     * explicit close/execute/reset calls.
     *
     * Using event for completion tracking improves performance by:
     * - Allowing async progress checking
     * - Enabling proper completion callbacks
     * - Supporting multiple outstanding operations
     */
    ret = zeCommandListAppendMemoryCopy(q_desc->cmd_list, dst, src,
                                        iov[0].length, event, 0, NULL);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("zeCommandListAppendMemoryCopy failed with error 0x%x", ret);
        goto err_cleanup;
    }

    clock_gettime(CLOCK_MONOTONIC, &start4);

    /* Store event info for progress tracking */
    event_desc->mapped_addr = mapped_addr;
    event_desc->comp        = comp;

    /* Add to active queue if this is the first event for this command list */
    if (ucs_queue_is_empty(&q_desc->event_queue)) {
        ucs_queue_push(&iface->active_queue, &q_desc->queue);
    }

    /* Push event to this command list's event queue */
    ucs_queue_push(&q_desc->event_queue, &event_desc->queue);

    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed_ms1 = (start2.tv_sec - start1.tv_sec) * 1000.0 +
    (start2.tv_nsec - start1.tv_nsec) / 1000000.0;

    elapsed_ms2 = (start3.tv_sec - start2.tv_sec) * 1000.0 +
    (start3.tv_nsec - start2.tv_nsec) / 1000000.0;

    elapsed_ms3 = (start4.tv_sec - start3.tv_sec) * 1000.0 +
    (start4.tv_nsec - start3.tv_nsec) / 1000000.0;

    elapsed_ms4 = (end.tv_sec - start4.tv_sec) * 1000.0 +
    (end.tv_nsec - start4.tv_nsec) / 1000000.0;

    elapsed_ms5 = (end.tv_sec - start1.tv_sec) * 1000.0 +
    (end.tv_nsec - start1.tv_nsec) / 1000000.0;

    ucs_trace("zeCommandListAppendMemoryCopy issued (async): cmd_list[%u/%u]=%p dst=%p src=%p len=%zu",
              cmd_list_idx, iface->num_cmd_lists, q_desc->cmd_list, dst, src, iov[0].length);
    ucs_info("uct_ze_ipc_post_copy projection: whole time cost is %.3f ms, open ipc time is %.3f, create event time is %.3f, copy time(host) is %.3f, other time is %.3f, total transfer size is %zu, cmd_list_idx=%u\n",
             elapsed_ms5, elapsed_ms1, elapsed_ms2, elapsed_ms3, elapsed_ms4, iov[0].length, cmd_list_idx);

    return UCS_INPROGRESS;

err_cleanup:
    uct_ze_ipc_free_event(iface, event, event_index);
    ucs_free(event_desc);
    uct_ze_ipc_unmap_memhandle(ep->remote_pid, key->address, mapped_addr,
                               iface->ze_context, local_fd,
                               iface->config.enable_cache);
    return UCS_ERR_IO_ERROR;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_ze_ipc_ep_get_zcopy,
                 (tl_ep, iov, iovcnt, remote_addr, rkey, comp),
                 uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                 uint64_t remote_addr, uct_rkey_t rkey,
                 uct_completion_t *comp)
{
    ucs_status_t status;

    ucs_info("ze_ipc_ep: GET_ZCOPY called remote_addr=0x%lx iovcnt=%zu total_len=%zu",
             (unsigned long)remote_addr, iovcnt, uct_iov_total_length(iov, iovcnt));

    status = uct_ze_ipc_post_copy(tl_ep, remote_addr, iov, rkey, comp,
                                  UCT_ZE_IPC_GET);
    if (UCS_STATUS_IS_ERR(status)) {
        ucs_error("ze_ipc_ep: GET_ZCOPY failed with status %s", ucs_status_string(status));
        return status;
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_ze_ipc_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                          uct_iov_total_length(iov, iovcnt));
    return status;
}


UCS_PROFILE_FUNC(ucs_status_t, uct_ze_ipc_ep_put_zcopy,
                 (tl_ep, iov, iovcnt, remote_addr, rkey, comp),
                 uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                 uint64_t remote_addr, uct_rkey_t rkey,
                 uct_completion_t *comp)
{
    ucs_status_t status;

    ucs_info("ze_ipc_ep: PUT_ZCOPY called remote_addr=0x%lx iovcnt=%zu total_len=%zu",
             (unsigned long)remote_addr, iovcnt, uct_iov_total_length(iov, iovcnt));

    status = uct_ze_ipc_post_copy(tl_ep, remote_addr, iov, rkey, comp,
                                  UCT_ZE_IPC_PUT);
    if (UCS_STATUS_IS_ERR(status)) {
        ucs_error("ze_ipc_ep: PUT_ZCOPY failed with status %s", ucs_status_string(status));
        return status;
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_ze_ipc_trace_data(remote_addr, rkey, "PUT_ZCOPY [length %zu]",
                          uct_iov_total_length(iov, iovcnt));
    return status;
}
