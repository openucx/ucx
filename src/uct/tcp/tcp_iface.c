/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcp.h"

#include <uct/base/uct_worker.h>
#include <ucs/async/async.h>
#include <ucs/sys/string.h>
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

  {"BACKLOG", "100",
   "Backlog size of incoming connections",
   ucs_offsetof(uct_tcp_iface_config_t, backlog), UCS_CONFIG_TYPE_UINT},

  {"MAX_POLL", "16",
   "Number of times to poll on a ready socket. 0 - no polling, -1 - until drained",
   ucs_offsetof(uct_tcp_iface_config_t, max_poll), UCS_CONFIG_TYPE_UINT},

  {"NODELAY", "y",
   "Set TCP_NODELAY socket option to disable Nagle algorithm. Setting this\n"
   "option usually provides better performance",
   ucs_offsetof(uct_tcp_iface_config_t, sockopt_nodelay), UCS_CONFIG_TYPE_BOOL},

  {"SNDBUF", "64k",
   "Socket send buffer size.",
   ucs_offsetof(uct_tcp_iface_config_t, sockopt_sndbuf), UCS_CONFIG_TYPE_MEMUNITS},

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
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);
    const in_addr_t *remote_inaddr = (const in_addr_t*)dev_addr;
    in_addr_t netmask = iface->config.netmask.sin_addr.s_addr;

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
                             UCT_IFACE_FLAG_AM_BCOPY         |
                             UCT_IFACE_FLAG_PENDING          |
                             UCT_IFACE_FLAG_CB_SYNC          |
                             UCT_IFACE_FLAG_EVENT_SEND_COMP  |
                             UCT_IFACE_FLAG_EVENT_RECV;

    attr->cap.am.max_bcopy = iface->config.buf_size - sizeof(uct_tcp_am_hdr_t);

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

unsigned uct_tcp_iface_progress(uct_iface_h tl_iface)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);
    struct epoll_event events[UCT_TCP_MAX_EVENTS];
    uct_tcp_ep_t *ep;
    unsigned count;
    int i, nevents;
    int max_events;

    ucs_trace_poll("iface=%p", iface);

    max_events = ucs_min(UCT_TCP_MAX_EVENTS, iface->config.max_poll);
    nevents = epoll_wait(iface->epfd, events, max_events, 0);
    if ((nevents < 0) && (errno != EINTR)) {
        ucs_error("epoll_wait(epfd=%d max=%d) failed: %m", iface->epfd,
                  max_events);
        return 0;
    }

    count = 0;
    for (i = 0; i < nevents; ++i) {
        ep = events[i].data.ptr;
        if (events[i].events & EPOLLIN) {
            count += uct_tcp_ep_progress_rx(ep);
        }
        if (events[i].events & EPOLLOUT) {
            count += uct_tcp_ep_progress_tx(ep);
        }
    }
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

static void uct_tcp_iface_connect_handler(int listen_fd, void *arg)
{
    uct_tcp_iface_t *iface = arg;
    struct sockaddr_in peer_addr;
    socklen_t addrlen;
    ucs_status_t status;
    uct_tcp_ep_t *ep;
    int fd;

    ucs_assert(listen_fd == iface->listen_fd);

    addrlen = sizeof(peer_addr);
    fd = accept(iface->listen_fd, (struct sockaddr*)&peer_addr, &addrlen);
    if (fd < 0) {
        if ((errno != EAGAIN) && (errno != EINTR)) {
            ucs_error("accept() failed: %m");
        }
        return;
    }

    ucs_debug("tcp_iface %p: accepted connection from %s:%d to fd %d", iface,
              inet_ntoa(peer_addr.sin_addr), ntohs(peer_addr.sin_port), fd);

    status = uct_tcp_ep_create(iface, fd, NULL, &ep);
    if (status != UCS_OK) {
        close(fd);
        return;
    }

    uct_tcp_ep_mod_events(ep, EPOLLIN, 0);
}

ucs_status_t uct_tcp_iface_set_sockopt(uct_tcp_iface_t *iface, int fd)
{
    int ret;

    ret = setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (void*)&iface->sockopt.nodelay,
                     sizeof(int));
    if (ret < 0) {
        ucs_error("Failed to set TCP_NODELAY on fd %d: %m", fd);
        return UCS_ERR_IO_ERROR;
    }

    ret = setsockopt(fd, SOL_SOCKET, SO_SNDBUF, (void*)&iface->sockopt.sndbuf,
                     sizeof(int));
    if (ret < 0) {
        ucs_error("Failed to set SO_SNDBUF on fd %d: %m", fd);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static uct_iface_ops_t uct_tcp_iface_ops = {
    .ep_am_bcopy              = uct_tcp_ep_am_bcopy,
    .ep_pending_add           = uct_tcp_ep_pending_add,
    .ep_pending_purge         = uct_tcp_ep_pending_purge,
    .ep_flush                 = uct_tcp_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create_connected      = uct_tcp_ep_create_connected,
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

static UCS_CLASS_INIT_FUNC(uct_tcp_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_tcp_iface_config_t *config = ucs_derived_of(tl_config, uct_tcp_iface_config_t);
    struct sockaddr_in bind_addr;
    ucs_status_t status;
    socklen_t addrlen;
    int ret;

    ucs_assert(params->open_mode & UCT_IFACE_OPEN_MODE_DEVICE);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_tcp_iface_ops, md, worker,
                              params, tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(params->mode.device.dev_name));

    ucs_strncpy_zero(self->if_name, params->mode.device.dev_name,
                     sizeof(self->if_name));
    self->outstanding            = 0;
    self->config.buf_size        = config->super.max_bcopy +
                                   sizeof(uct_tcp_am_hdr_t);
    self->config.prefer_default  = config->prefer_default;
    self->config.max_poll        = config->max_poll;
    self->sockopt.nodelay        = config->sockopt_nodelay;
    self->sockopt.sndbuf         = config->sockopt_sndbuf;
    ucs_list_head_init(&self->ep_list);

    status = uct_tcp_netif_inaddr(self->if_name, &self->config.ifaddr,
                                  &self->config.netmask);
    if (status != UCS_OK) {
        goto err;
    }

    self->epfd = epoll_create(1);
    if (self->epfd < 0) {
        ucs_error("epoll_create() failed: %m");
        goto err;
    }

    /* Create the server socket for accepting incoming connections */
    status = ucs_tcpip_socket_create(&self->listen_fd);
    if (status != UCS_OK) {
        goto err_close_epfd;
    }

    /* Set the server socket to non-blocking mode */
    status = ucs_sys_fcntl_modfl(self->listen_fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_close_sock;
    }

    /* Bind socket to random available port */
    bind_addr = self->config.ifaddr;
    bind_addr.sin_port = 0;
    ret = bind(self->listen_fd, (struct sockaddr *)&bind_addr, sizeof(bind_addr));
    if (ret < 0) {
        ucs_error("bind() failed: %m");
        goto err_close_sock;
    }

    /* Get the port which was selected for the socket */
    addrlen = sizeof(bind_addr);
    ret = getsockname(self->listen_fd, (struct sockaddr*)&bind_addr, &addrlen);
    if (ret < 0) {
        ucs_error("getsockname(fd=%d) failed: %m", self->listen_fd);
        goto err_close_sock;
    }
    self->config.ifaddr.sin_port = bind_addr.sin_port;

    /* Listen for connections */
    ret = listen(self->listen_fd, config->backlog);
    if (ret < 0) {
        ucs_error("listen(backlog=%d)", config->backlog);
        status = UCS_ERR_IO_ERROR;
        goto err_close_sock;
    }

    ucs_debug("tcp_iface %p: listening for connections on %s:%d", self,
              inet_ntoa(bind_addr.sin_addr), ntohs(bind_addr.sin_port));

    /* Register event handler for incoming connections */
    status = ucs_async_set_event_handler(self->super.worker->async->mode,
                                         self->listen_fd, POLLIN|POLLERR,
                                         uct_tcp_iface_connect_handler, self,
                                         self->super.worker->async);
    if (status != UCS_OK) {
        goto err_close_sock;
    }

    return UCS_OK;

err_close_sock:
    close(self->listen_fd);
err_close_epfd:
    close(self->epfd);
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

    close(self->listen_fd);
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

