/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcp.h"

#include <ucs/async/async.h>
#include <sys/socket.h>
#include <sys/poll.h>


static ucs_config_field_t uct_tcp_iface_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_tcp_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"BACKLOG", "100",
     "Backlog size of incoming connections",
     ucs_offsetof(uct_tcp_iface_config_t, backlog), UCS_CONFIG_TYPE_UINT},

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

static ucs_status_t uct_tcp_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *attr)
{
    memset(attr, 0, sizeof(*attr));
    attr->iface_addr_len   = sizeof(in_port_t);
    attr->device_addr_len  = sizeof(struct in_addr);
    attr->ep_addr_len      = 0;
    attr->cap.flags        = UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    attr->latency          = 50e-6;       /* 50 usec */
    attr->bandwidth        = 100e6 / 8.0; /* 100 Mbit/s */
    attr->overhead         = 50e-6;       /* 50 usec */
    attr->priority         = 0;
    return UCS_OK;
}

static void uct_tcp_iface_connect_handler(int fd, void *arg)
{
    uct_tcp_iface_t *iface = arg;
    int sockfd;

    sockfd = accept(iface->listen_fd, NULL, NULL);
    if (sockfd < 0) {
        if (errno != EAGAIN) {
            ucs_error("accept() failed: %m");
        }
        return;
    }

    close(sockfd); /* TODO accept data from the connection */
}

static uct_iface_ops_t uct_tcp_iface_ops = {
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_tcp_iface_t),
    .iface_get_device_address = uct_tcp_iface_get_device_address,
    .iface_get_address        = uct_tcp_iface_get_address,
    .iface_query              = uct_tcp_iface_query,
    .iface_is_reachable       = (void*)ucs_empty_function_return_inprogress, /* TODO */
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_tcp_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_tcp_ep_t),
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
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

    status = uct_tcp_netif_inaddr(params->dev_name, &self->config.ifaddr);
    if (status != UCS_OK) {
        goto err;
    }

    /* Create the server socket for accepting incoming connections */
    status = uct_tcp_socket_create(&self->listen_fd);
    if (status != UCS_OK) {
        goto err;
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
    close(self->listen_fd);
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
    unsigned num_resources;
    struct if_nameindex *ifs, *netif;
    ucs_status_t status;

    ifs = if_nameindex();
    if (ifs == NULL) {
        ucs_error("if_nameindex() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    resources     = NULL;
    num_resources = 0;
    for (netif = ifs; netif->if_name != NULL; ++netif) {
        if (!uct_tcp_netif_check(netif->if_name)) {
            continue;
        }

        tmp = ucs_realloc(resources, sizeof(*resources) * (num_resources + 1),
                          "resource desc");
        if (tmp == NULL) {
            ucs_free(resources);
            status = UCS_ERR_NO_MEMORY;
            goto out_free_ifaddrs;
        }
        resources = tmp;

        resource = &resources[num_resources++];
        ucs_snprintf_zero(resource->tl_name, sizeof(resource->tl_name),
                          "%s", UCT_TCP_NAME);
        ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name),
                          "%s", netif->if_name);
        resource->dev_type = UCT_DEVICE_TYPE_NET;
    }

    *num_resources_p = num_resources;
    *resource_p      = resources;
    status           = UCS_OK;

out_free_ifaddrs:
    if_freenameindex(ifs);
out:
    return status;
}

UCT_TL_COMPONENT_DEFINE(uct_tcp_tl, uct_tcp_query_tl_resources, uct_tcp_iface_t,
                        UCT_TCP_NAME, "TCP_", uct_tcp_iface_config_table,
                        uct_tcp_iface_config_t);
UCT_MD_REGISTER_TL(&uct_tcp_md, &uct_tcp_tl);
