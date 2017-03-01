/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcp.h"

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

  {"TCP_NODELAY", "y",
   "Set TCP_NODELAY socket option to disable Nagle algorithm. Setting this\n"
   "option usually provides better performance",
   ucs_offsetof(uct_tcp_iface_config_t, sockopt_nodelay), UCS_CONFIG_TYPE_BOOL},

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
                             UCT_IFACE_FLAG_PENDING;

    attr->cap.am.max_bcopy = iface->config.max_bcopy;

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

static void uct_tcp_iface_connect_handler(int fd, void *arg)
{
    uct_tcp_iface_t *iface = arg;
    struct sockaddr_in client_addr;
    socklen_t client_addrlen;
    int sockfd;

    ucs_assert(fd == iface->listen_fd);

    memset(&client_addr, 0, sizeof(client_addr));
    client_addrlen = sizeof(client_addr);
    sockfd = accept(iface->listen_fd, (struct sockaddr*)&client_addr,
                    &client_addrlen);
    if (sockfd < 0) {
        if (errno != EAGAIN) {
            ucs_error("accept() failed: %m");
        }
        return;
    }

    ucs_trace("accepted connection from %s:%d", inet_ntoa(client_addr.sin_addr),
              ntohs(client_addr.sin_port));
    uct_tcp_iface_connection_accepted(iface, sockfd);
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

    return UCS_OK;
}

static uct_iface_ops_t uct_tcp_iface_ops = {
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_tcp_iface_t),
    .iface_get_device_address = uct_tcp_iface_get_device_address,
    .iface_get_address        = uct_tcp_iface_get_address,
    .iface_query              = uct_tcp_iface_query,
    .iface_is_reachable       = uct_tcp_iface_is_reachable,
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_tcp_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_tcp_ep_t),
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
};

static ucs_mpool_ops_t uct_tcp_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_mmap,
    .chunk_release = ucs_mpool_chunk_munmap,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
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

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_tcp_iface_ops, md, worker,
                              tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(params->dev_name));

    ucs_strncpy_zero(self->if_name, params->dev_name, sizeof(self->if_name));
    self->config.max_bcopy       = config->super.max_bcopy;
    self->config.prefer_default  = config->prefer_default;
    self->sockopt.nodelay        = config->sockopt_nodelay;

    kh_init_inplace(uct_tcp_fd_hash, &self->fd_hash);

    status = uct_tcp_netif_inaddr(self->if_name, &self->config.ifaddr,
                                  &self->config.netmask);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_mpool_init(&self->mp, 0,
                            self->config.max_bcopy,
                            0,                        /* alignment offset */
                            UCS_SYS_CACHE_LINE_SIZE,  /* alignment */
                            32,                       /* grow */
                            -1,                       /* max buffers */
                            &uct_tcp_mpool_ops,
                            "tcp_desc");
    if (status != UCS_OK) {
        goto err;
    }

    /* Create the server socket for accepting incoming connections */
    status = uct_tcp_socket_create(&self->listen_fd);
    if (status != UCS_OK) {
        goto err_mpool_cleanup;
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

    ucs_debug("listening for connections on %s:%d", inet_ntoa(bind_addr.sin_addr),
              ntohs(bind_addr.sin_port));

    /* Register event handler for incoming connections */
    status = ucs_async_set_event_handler(worker->async->mode, self->listen_fd,
                                         POLLIN|POLLERR,
                                         uct_tcp_iface_connect_handler, self,
                                         worker->async);
    if (status != UCS_OK) {
        goto err_close_sock;
    }

    return UCS_OK;

err_close_sock:
    close(self->listen_fd);
err_mpool_cleanup:
    ucs_mpool_cleanup(&self->mp, 0);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_tcp_iface_t)
{
    ucs_status_t status;

    status = ucs_async_remove_handler(self->listen_fd, 1);
    if (status != UCS_OK) {
        ucs_warn("failed to remove handler for server socket fd=%d", self->listen_fd);
    }

    uct_tcp_iface_recv_cleanup(self);
    close(self->listen_fd);
    ucs_mpool_cleanup(&self->mp, 1);
    kh_destroy_inplace(uct_tcp_fd_hash, &self->fd_hash);
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

        if (!uct_tcp_netif_check(entry->d_name)) {
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
