/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcp.h"

#include <ucs/async/async.h>
#include <ucs/sys/string.h>
#include <ucs/config/types.h>
#include <sys/socket.h>
#include <sys/poll.h>
#include <netinet/tcp.h>
#include <dirent.h>

static ucs_config_field_t uct_tcp_iface_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_tcp_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

  {"PREFER_DEFAULT", "y",
   "Give higher priority to the default network interface on the host",
   ucs_offsetof(uct_tcp_iface_config_t, prefer_default), UCS_CONFIG_TYPE_BOOL},

  {"MAX_POLL", UCS_PP_MAKE_STRING(UCT_TCP_MAX_EVENTS),
   "Number of times to poll on a ready socket. 0 - no polling, -1 - until drained",
   ucs_offsetof(uct_tcp_iface_config_t, max_poll), UCS_CONFIG_TYPE_UINT},

  {"NODELAY", "y",
   "Set TCP_NODELAY socket option to disable Nagle algorithm. Setting this\n"
   "option usually provides better performance",
   ucs_offsetof(uct_tcp_iface_config_t, sockopt_nodelay), UCS_CONFIG_TYPE_BOOL},

  {"SNDBUF", "64k",
   "Socket send buffer size",
   ucs_offsetof(uct_tcp_iface_config_t, sockopt_sndbuf), UCS_CONFIG_TYPE_MEMUNITS},

  {"RCVBUF", "auto",
   "Socket receive buffer size",
   ucs_offsetof(uct_tcp_iface_config_t, sockopt_rcvbuf), UCS_CONFIG_TYPE_MEMUNITS},

  UCT_IFACE_MPOOL_CONFIG_FIELDS("TX_", -1, 8, "send",
                                ucs_offsetof(uct_tcp_iface_config_t, tx_mpool), ""),

  UCT_IFACE_MPOOL_CONFIG_FIELDS("RX_", -1, 8, "receive",
                                ucs_offsetof(uct_tcp_iface_config_t, rx_mpool), ""),

  {NULL}
};


static UCS_CLASS_DEFINE_DELETE_FUNC(uct_tcp_iface_t, uct_iface_t);

static ucs_status_t uct_tcp_iface_get_device_address(uct_iface_h tl_iface,
                                                     uct_device_addr_t *addr)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);

    *(struct in_addr*)addr = iface->config.ifaddr.sin_addr;
    return UCS_OK;
}

static ucs_status_t uct_tcp_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *addr)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);

    *(in_port_t*)addr = iface->config.ifaddr.sin_port;
    return UCS_OK;
}

static int uct_tcp_iface_is_reachable(const uct_iface_h tl_iface,
                                      const uct_device_addr_t *dev_addr,
                                      const uct_iface_addr_t *iface_addr)
{
    uct_tcp_iface_t *iface         = ucs_derived_of(tl_iface, uct_tcp_iface_t);
    const in_addr_t *remote_inaddr = (const in_addr_t*)dev_addr;
    in_addr_t netmask              = iface->config.netmask.sin_addr.s_addr;

    return (*remote_inaddr & netmask) ==
           (iface->config.ifaddr.sin_addr.s_addr & netmask);
}

static ucs_status_t uct_tcp_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *attr)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);
    ucs_status_t status;
    int is_default;

    memset(attr, 0, sizeof(*attr));
    attr->iface_addr_len   = sizeof(in_port_t);
    attr->device_addr_len  = sizeof(struct in_addr);
    attr->cap.flags        = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                             UCT_IFACE_FLAG_AM_SHORT         |
                             UCT_IFACE_FLAG_AM_BCOPY         |
                             UCT_IFACE_FLAG_PENDING          |
                             UCT_IFACE_FLAG_CB_SYNC          |
                             UCT_IFACE_FLAG_EVENT_SEND_COMP  |
                             UCT_IFACE_FLAG_EVENT_RECV;
    attr->cap.flags       += UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;
    attr->cap.am.max_short = attr->cap.am.max_bcopy =
        iface->am_buf_size - sizeof(uct_tcp_am_hdr_t);

    status = uct_tcp_netif_caps(iface->if_name, &attr->latency.overhead,
                                &attr->bandwidth);
    if (status != UCS_OK) {
        return status;
    }

    attr->latency.growth   = 0;
    attr->overhead         = 50e-6;  /* 50 usec */

    if (iface->config.prefer_default) {
        status = uct_tcp_netif_is_default(iface->if_name, &is_default);
        if (status != UCS_OK) {
             return status;
        }

        attr->priority = is_default ? 0 : 1;
    } else {
        attr->priority = 0;
    }

    return UCS_OK;
}

static ucs_status_t uct_tcp_iface_event_fd_get(uct_iface_h tl_iface, int *fd_p)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);

    *fd_p = iface->epfd;
    return UCS_OK;
}

static inline unsigned
uct_tcp_iface_handle_events(uct_tcp_ep_t *ep, uint32_t epoll_events)
{
    unsigned count = 0;

    if (epoll_events & EPOLLIN) {
        count += uct_tcp_ep_progress(ep, UCT_TCP_EP_CTX_TYPE_RX);
    }
    if (epoll_events & EPOLLOUT) {
        count += uct_tcp_ep_progress(ep, UCT_TCP_EP_CTX_TYPE_TX);
    }

    return count;
}

unsigned uct_tcp_iface_progress(uct_iface_h tl_iface)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);
    unsigned read_events   = 0;
    unsigned count         = 0;
    struct epoll_event events[UCT_TCP_MAX_EVENTS];
    int i, nevents, max_events;

    do {
        max_events = ucs_min(iface->config.max_poll - read_events,
                             UCT_TCP_MAX_EVENTS);

        nevents = epoll_wait(iface->epfd, events, max_events, 0);
        if (ucs_unlikely((nevents < 0))) {
            if (errno == EINTR) {
                /* force a new loop iteration */
                nevents = max_events;
                continue;
            }
            ucs_error("epoll_wait(epfd=%d max=%d) failed: %m",
                      iface->epfd, max_events);
            return 0;
        }

        for (i = 0; i < nevents; ++i) {
            count += uct_tcp_iface_handle_events(events[i].data.ptr,
                                                 events[i].events);
        }

        read_events += nevents;

        ucs_trace_poll("iface=%p epoll_wait()=%d, total=%u",
                       iface, nevents, read_events);
    } while ((read_events < iface->config.max_poll) && (nevents == max_events));

    return count;
}

static ucs_status_t uct_tcp_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                        uct_completion_t *comp)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (iface->outstanding) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super);
        return UCS_INPROGRESS;
    }

    UCT_TL_IFACE_STAT_FLUSH(&iface->super);
    return UCS_OK;
}

void uct_tcp_iface_outstanding_inc(uct_tcp_iface_t *iface)
{
    iface->outstanding++;
}

void uct_tcp_iface_outstanding_dec(uct_tcp_iface_t *iface)
{
    ucs_assert(iface->outstanding > 0);
    iface->outstanding--;
}

static void uct_tcp_iface_listen_close(uct_tcp_iface_t *iface)
{
    if (iface->listen_fd != -1) {
        close(iface->listen_fd);
        iface->listen_fd = -1;
    }
}

static void uct_tcp_iface_connect_handler(int listen_fd, void *arg)
{
    uct_tcp_iface_t *iface = arg;
    struct sockaddr_in peer_addr;
    socklen_t addrlen;
    ucs_status_t status;
    int fd;

    ucs_assert(listen_fd == iface->listen_fd);

    for (;;) {
        addrlen = sizeof(peer_addr);

        fd = accept(iface->listen_fd, (struct sockaddr*)&peer_addr, &addrlen);
        if (fd < 0) {
            if ((errno != EAGAIN) && (errno != EWOULDBLOCK) && (errno != EINTR)) {
                ucs_error("accept() failed: %m");
                uct_tcp_iface_listen_close(iface);
            }
            return;
        }

        status = uct_tcp_cm_handle_incoming_conn(iface, &peer_addr, fd);
        if (status != UCS_OK) {
            close(fd);
            return;
        }
    }
}

ucs_status_t uct_tcp_iface_set_sockopt(uct_tcp_iface_t *iface, int fd)
{
    ucs_status_t status;

    status = ucs_socket_setopt(fd, IPPROTO_TCP, TCP_NODELAY,
                               (const void*)&iface->sockopt.nodelay,
                               sizeof(int));
    if (status != UCS_OK) {
        return status;
    }

    if (iface->sockopt.sndbuf != UCS_MEMUNITS_AUTO) {
        status = ucs_socket_setopt(fd, SOL_SOCKET, SO_SNDBUF,
                                   (const void*)&iface->sockopt.sndbuf,
                                   sizeof(int));
        if (status != UCS_OK) {
            return status;
        }
    }

    if (iface->sockopt.rcvbuf != UCS_MEMUNITS_AUTO) {
        status = ucs_socket_setopt(fd, SOL_SOCKET, SO_RCVBUF,
                                   (const void*)&iface->sockopt.rcvbuf,
                                   sizeof(int));
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static uct_iface_ops_t uct_tcp_iface_ops = {
    .ep_am_short              = uct_tcp_ep_am_short,
    .ep_am_bcopy              = uct_tcp_ep_am_bcopy,
    .ep_pending_add           = uct_tcp_ep_pending_add,
    .ep_pending_purge         = uct_tcp_ep_pending_purge,
    .ep_flush                 = uct_tcp_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = uct_tcp_ep_create,
    .ep_destroy               = uct_tcp_ep_destroy,
    .iface_flush              = uct_tcp_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_tcp_iface_progress,
    .iface_event_fd_get       = uct_tcp_iface_event_fd_get,
    .iface_event_arm          = ucs_empty_function_return_success,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_tcp_iface_t),
    .iface_query              = uct_tcp_iface_query,
    .iface_get_address        = uct_tcp_iface_get_address,
    .iface_get_device_address = uct_tcp_iface_get_device_address,
    .iface_is_reachable       = uct_tcp_iface_is_reachable
};

static ucs_status_t uct_tcp_iface_listener_init(uct_tcp_iface_t *iface)
{
    struct sockaddr_in bind_addr = iface->config.ifaddr;
    socklen_t addrlen            = sizeof(bind_addr);
    int backlog                  = ucs_socket_max_conn();
    ucs_status_t status;
    int ret;

    /* Create the server socket for accepting incoming connections */
    status = ucs_socket_create(AF_INET, SOCK_STREAM, &iface->listen_fd);
    if (status != UCS_OK) {
        return status;
    }

    /* Set the server socket to non-blocking mode */
    status = ucs_sys_fcntl_modfl(iface->listen_fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_close_sock;
    }

    /* Bind socket to random available port */
    bind_addr.sin_port = 0;
    ret = bind(iface->listen_fd, (struct sockaddr*)&bind_addr, sizeof(bind_addr));
    if (ret < 0) {
        ucs_error("bind(fd=%d) failed: %m", iface->listen_fd);
        status = UCS_ERR_IO_ERROR;
        goto err_close_sock;
    }

    /* Get the port which was selected for the socket */
    ret = getsockname(iface->listen_fd, (struct sockaddr*)&bind_addr, &addrlen);
    if (ret < 0) {
        ucs_error("getsockname(fd=%d) failed: %m", iface->listen_fd);
        status = UCS_ERR_IO_ERROR;
        goto err_close_sock;
    }
    iface->config.ifaddr.sin_port = bind_addr.sin_port;

    /* Listen for connections */
    ret = listen(iface->listen_fd, backlog);
    if (ret < 0) {
        ucs_error("listen(fd=%d; backlog=%d)",
                  iface->listen_fd, backlog);
        status = UCS_ERR_IO_ERROR;
        goto err_close_sock;
    }

    ucs_debug("tcp_iface %p: listening for connections on %s:%d", iface,
              inet_ntoa(bind_addr.sin_addr), ntohs(bind_addr.sin_port));

    /* Register event handler for incoming connections */
    status = ucs_async_set_event_handler(iface->super.worker->async->mode,
                                         iface->listen_fd, POLLIN|POLLERR,
                                         uct_tcp_iface_connect_handler, iface,
                                         iface->super.worker->async);
    if (status != UCS_OK) {
        goto err_close_sock;
    }

    return UCS_OK;

err_close_sock:
    close(iface->listen_fd);
    return status;
}

static ucs_mpool_ops_t uct_tcp_mpool_ops = {
    ucs_mpool_chunk_malloc,
    ucs_mpool_chunk_free,
    NULL,
    NULL
};

static UCS_CLASS_INIT_FUNC(uct_tcp_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_tcp_iface_config_t *config = ucs_derived_of(tl_config,
                                                    uct_tcp_iface_config_t);
    ucs_status_t status;

    UCT_CHECK_PARAM(params->field_mask & UCT_IFACE_PARAM_FIELD_OPEN_MODE,
                    "UCT_IFACE_PARAM_FIELD_OPEN_MODE is not defined");
    if (!(params->open_mode & UCT_IFACE_OPEN_MODE_DEVICE)) {
        ucs_error("only UCT_IFACE_OPEN_MODE_DEVICE is supported");
        return UCS_ERR_UNSUPPORTED;
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_tcp_iface_ops, md, worker,
                              params, tl_config
                              UCS_STATS_ARG((params->field_mask &
                                             UCT_IFACE_PARAM_FIELD_STATS_ROOT) ?
                                            params->stats_root : NULL)
                              UCS_STATS_ARG(params->mode.device.dev_name));

    ucs_strncpy_zero(self->if_name, params->mode.device.dev_name,
                     sizeof(self->if_name));
    self->outstanding           = 0;
    self->config.prefer_default = config->prefer_default;
    self->config.max_poll       = config->max_poll;
    self->sockopt.nodelay       = config->sockopt_nodelay;
    self->sockopt.sndbuf        = config->sockopt_sndbuf;
    self->sockopt.rcvbuf        = config->sockopt_rcvbuf;
    ucs_list_head_init(&self->ep_list);

    self->am_buf_size = config->super.max_bcopy + sizeof(uct_tcp_am_hdr_t);

    status = ucs_mpool_init(&self->tx_mpool, 0, self->am_buf_size,
                            0, UCS_SYS_CACHE_LINE_SIZE,
                            (config->tx_mpool.bufs_grow == 0) ?
                            32 : config->tx_mpool.bufs_grow,
                            config->tx_mpool.max_bufs,
                            &uct_tcp_mpool_ops, "uct_tcp_iface_tx_buf_mp");
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_mpool_init(&self->rx_mpool, 0, self->am_buf_size * 2,
                            0, UCS_SYS_CACHE_LINE_SIZE,
                            (config->rx_mpool.bufs_grow == 0) ?
                            32 : config->rx_mpool.bufs_grow,
                            config->rx_mpool.max_bufs,
                            &uct_tcp_mpool_ops, "uct_tcp_iface_rx_buf_mp");
    if (status != UCS_OK) {
        goto err_cleanup_tx_mpool;
    }

    if (ucs_derived_of(worker, uct_priv_worker_t)->thread_mode == UCS_THREAD_MODE_MULTI) {
        ucs_error("TCP transport does not support multi-threaded worker");
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_tcp_netif_inaddr(self->if_name, &self->config.ifaddr,
                                  &self->config.netmask);
    if (status != UCS_OK) {
        goto err_cleanup_rx_mpool;
    }

    self->epfd = epoll_create(1);
    if (self->epfd < 0) {
        ucs_error("epoll_create() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_cleanup_rx_mpool;
    }

    status = uct_tcp_iface_listener_init(self);
    if (status != UCS_OK) {
        goto err_close_epfd;
    }

    return UCS_OK;

err_close_epfd:
    close(self->epfd);
err_cleanup_rx_mpool:
    ucs_mpool_cleanup(&self->rx_mpool, 1);
err_cleanup_tx_mpool:
    ucs_mpool_cleanup(&self->tx_mpool, 1);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_tcp_iface_t)
{
    uct_tcp_ep_t *ep, *tmp;
    ucs_status_t status;

    ucs_debug("tcp_iface %p: destroying", self);

    uct_base_iface_progress_disable(&self->super.super, UCT_PROGRESS_SEND|
                                                        UCT_PROGRESS_RECV);

    status = ucs_async_remove_handler(self->listen_fd, 1);
    if (status != UCS_OK) {
        ucs_warn("failed to remove handler for server socket fd=%d", self->listen_fd);
    }

    ucs_list_for_each_safe(ep, tmp, &self->ep_list, list) {
        uct_tcp_ep_destroy(&ep->super.super);
    }

    ucs_mpool_cleanup(&self->rx_mpool, 1);
    ucs_mpool_cleanup(&self->tx_mpool, 1);

    uct_tcp_iface_listen_close(self);
    close(self->epfd);
}

UCS_CLASS_DEFINE(uct_tcp_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_tcp_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static ucs_status_t uct_tcp_query_tl_resources(uct_md_h md,
                                               uct_tl_resource_desc_t **resource_p,
                                               unsigned *num_resources_p)
{
    uct_tl_resource_desc_t *resources, *tmp, *resource;
    static const char *netdev_dir = "/sys/class/net";
    struct dirent *entry;
    unsigned num_resources;
    ucs_status_t status;
    DIR *dir;

    dir = opendir(netdev_dir);
    if (dir == NULL) {
        ucs_error("opendir(%s) failed: %m", netdev_dir);
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    resources     = NULL;
    num_resources = 0;
    for (;;) {
        errno = 0;
        entry = readdir(dir);
        if (entry == NULL) {
            if (errno != 0) {
                ucs_error("readdir(%s) failed: %m", netdev_dir);
                ucs_free(resources);
                status = UCS_ERR_IO_ERROR;
                goto out_closedir;
            }
            break; /* no more items */
        }

        if (!ucs_netif_is_active(entry->d_name)) {
            continue;
        }

        tmp = ucs_realloc(resources, sizeof(*resources) * (num_resources + 1),
                          "resource desc");
        if (tmp == NULL) {
            ucs_free(resources);
            status = UCS_ERR_NO_MEMORY;
            goto out_closedir;
        }
        resources = tmp;

        resource = &resources[num_resources++];
        ucs_snprintf_zero(resource->tl_name, sizeof(resource->tl_name),
                          "%s", UCT_TCP_NAME);
        ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name),
                          "%s", entry->d_name);
        resource->dev_type = UCT_DEVICE_TYPE_NET;
    }

    *num_resources_p = num_resources;
    *resource_p      = resources;
    status           = UCS_OK;

out_closedir:
    closedir(dir);
out:
    return status;
}

UCT_TL_COMPONENT_DEFINE(uct_tcp_tl, uct_tcp_query_tl_resources, uct_tcp_iface_t,
                        UCT_TCP_NAME, "TCP_", uct_tcp_iface_config_table,
                        uct_tcp_iface_config_t);
UCT_MD_REGISTER_TL(&uct_tcp_md, &uct_tcp_tl);

