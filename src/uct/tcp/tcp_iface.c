/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "tcp.h"

#include <ucs/async/async.h>
#include <ucs/sys/string.h>
#include <ucs/config/types.h>
#include <sys/socket.h>
#include <sys/poll.h>
#include <netinet/tcp.h>
#include <dirent.h>


extern ucs_class_t UCS_CLASS_DECL_NAME(uct_tcp_iface_t);

static ucs_config_field_t uct_tcp_iface_config_table[] = {
  {"", "MAX_NUM_EPS=256", NULL,
   ucs_offsetof(uct_tcp_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

  {"TX_SEG_SIZE", "8kb",
   "Size of send copy-out buffer",
   ucs_offsetof(uct_tcp_iface_config_t, tx_seg_size), UCS_CONFIG_TYPE_MEMUNITS},
  
  {"RX_SEG_SIZE", "64kb",
   "Size of receive copy-out buffer",
   ucs_offsetof(uct_tcp_iface_config_t, rx_seg_size), UCS_CONFIG_TYPE_MEMUNITS},

  {"MAX_IOV", "6",
   "Maximum IOV count that can contain user-defined payload in a single\n"
   "call to non-blocking vector socket send",
   ucs_offsetof(uct_tcp_iface_config_t, max_iov), UCS_CONFIG_TYPE_ULONG},

  {"SENDV_THRESH", "2kb",
   "Threshold for switching from send() to sendmsg() for short active messages",
   ucs_offsetof(uct_tcp_iface_config_t, sendv_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"PREFER_DEFAULT", "y",
   "Give higher priority to the default network interface on the host",
   ucs_offsetof(uct_tcp_iface_config_t, prefer_default), UCS_CONFIG_TYPE_BOOL},

  {"PUT_ENABLE", "y",
   "Enable PUT Zcopy support",
   ucs_offsetof(uct_tcp_iface_config_t, put_enable), UCS_CONFIG_TYPE_BOOL},

  {"CONN_NB", "n",
   "Enable non-blocking connection establishment. It may improve startup "
   "time, but can lead to connection resets due to high load on TCP/IP stack",
   ucs_offsetof(uct_tcp_iface_config_t, conn_nb), UCS_CONFIG_TYPE_BOOL},

  {"MAX_POLL", UCS_PP_MAKE_STRING(UCT_TCP_MAX_EVENTS),
   "Number of times to poll on a ready socket. 0 - no polling, -1 - until drained",
   ucs_offsetof(uct_tcp_iface_config_t, max_poll), UCS_CONFIG_TYPE_UINT},

  {UCT_TCP_CONFIG_MAX_CONN_RETRIES, "25",
   "How many connection establishment attempts should be done if dropped "
   "connection was detected due to lack of system resources",
   ucs_offsetof(uct_tcp_iface_config_t, max_conn_retries), UCS_CONFIG_TYPE_UINT},

  {"NODELAY", "y",
   "Set TCP_NODELAY socket option to disable Nagle algorithm. Setting this\n"
   "option usually provides better performance",
   ucs_offsetof(uct_tcp_iface_config_t, sockopt_nodelay), UCS_CONFIG_TYPE_BOOL},

  UCT_TCP_SEND_RECV_BUF_FIELDS(ucs_offsetof(uct_tcp_iface_config_t, sockopt)),

  UCT_TCP_SYN_CNT(ucs_offsetof(uct_tcp_iface_config_t, syn_cnt)),

  UCT_IFACE_MPOOL_CONFIG_FIELDS("TX_", -1, 8, "send",
                                ucs_offsetof(uct_tcp_iface_config_t, tx_mpool), ""),

  UCT_IFACE_MPOOL_CONFIG_FIELDS("RX_", -1, 8, "receive",
                                ucs_offsetof(uct_tcp_iface_config_t, rx_mpool), ""),

  {"PORT_RANGE", "0",
   "Generate a random TCP port number from that range. A value of zero means\n "
   "let the operating system select the port number.",
   ucs_offsetof(uct_tcp_iface_config_t, port_range), UCS_CONFIG_TYPE_RANGE_SPEC},

  {NULL}
};


static UCS_CLASS_DEFINE_DELETE_FUNC(uct_tcp_iface_t, uct_iface_t);

static ucs_status_t uct_tcp_iface_get_device_address(uct_iface_h tl_iface,
                                                     uct_device_addr_t *addr)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);

    *(struct sockaddr_in*)addr = iface->config.ifaddr;
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
    /* We always report that a peer is reachable. connect() call will
     * fail if the peer is unreachable when creating UCT/TCP EP */
    return 1;
}

static ucs_status_t uct_tcp_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *attr)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);
    size_t am_buf_size     = iface->config.tx_seg_size - sizeof(uct_tcp_am_hdr_t);
    ucs_status_t status;
    int is_default;

    uct_base_iface_query(&iface->super, attr);

    status = uct_tcp_netif_caps(iface->if_name, &attr->latency.c,
                                &attr->bandwidth.shared);
    if (status != UCS_OK) {
        return status;
    }

    attr->iface_addr_len   = sizeof(in_port_t);
    attr->device_addr_len  = sizeof(struct sockaddr_in);
    attr->cap.flags        = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                             UCT_IFACE_FLAG_AM_SHORT         |
                             UCT_IFACE_FLAG_AM_BCOPY         |
                             UCT_IFACE_FLAG_PENDING          |
                             UCT_IFACE_FLAG_CB_SYNC          |
                             UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;
    attr->cap.event_flags  = UCT_IFACE_FLAG_EVENT_SEND_COMP |
                             UCT_IFACE_FLAG_EVENT_RECV      |
                             UCT_IFACE_FLAG_EVENT_FD;

    attr->cap.am.max_short = am_buf_size;
    attr->cap.am.max_bcopy = am_buf_size;

    if (iface->config.zcopy.max_iov > UCT_TCP_EP_ZCOPY_SERVICE_IOV_COUNT) {
        /* AM */
        attr->cap.am.max_iov          = iface->config.zcopy.max_iov -
                                        UCT_TCP_EP_ZCOPY_SERVICE_IOV_COUNT;
        attr->cap.am.max_zcopy        = iface->config.rx_seg_size -
                                        sizeof(uct_tcp_am_hdr_t);
        attr->cap.am.max_hdr          = iface->config.zcopy.max_hdr;
        attr->cap.am.opt_zcopy_align  = 1;
        attr->cap.flags              |= UCT_IFACE_FLAG_AM_ZCOPY;

        if (iface->config.put_enable) {
            /* PUT */
            attr->cap.put.max_iov          = iface->config.zcopy.max_iov -
                                             UCT_TCP_EP_ZCOPY_SERVICE_IOV_COUNT;
            attr->cap.put.max_zcopy        = UCT_TCP_EP_PUT_ZCOPY_MAX -
                                             UCT_TCP_EP_PUT_SERVICE_LENGTH;
            attr->cap.put.opt_zcopy_align  = 1;
            attr->cap.flags               |= UCT_IFACE_FLAG_PUT_ZCOPY;
        }
    }

    attr->bandwidth.dedicated = 0;
    attr->latency.m           = 0;
    attr->overhead            = 50e-6;  /* 50 usec */

    if (iface->config.prefer_default) {
        status = uct_tcp_netif_is_default(iface->if_name, &is_default);
        if (status != UCS_OK) {
             return status;
        }

        attr->priority    = is_default ? 0 : 1;
    } else {
        attr->priority    = 0;
    }

    return UCS_OK;
}

static ucs_status_t uct_tcp_iface_event_fd_get(uct_iface_h tl_iface, int *fd_p)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);

    return ucs_event_set_fd_get(iface->event_set, fd_p);
}

static void uct_tcp_iface_handle_events(void *callback_data,
                                        int events, void *arg)
{
    unsigned *count  = (unsigned*)arg;
    uct_tcp_ep_t *ep = (uct_tcp_ep_t*)callback_data;

    ucs_assertv(ep->conn_state != UCT_TCP_EP_CONN_STATE_CLOSED, "ep=%p", ep);

    if (events & UCS_EVENT_SET_EVREAD) {
        *count += uct_tcp_ep_cm_state[ep->conn_state].rx_progress(ep);
    }
    if (events & UCS_EVENT_SET_EVWRITE) {
        *count += uct_tcp_ep_cm_state[ep->conn_state].tx_progress(ep);
    }
}

unsigned uct_tcp_iface_progress(uct_iface_h tl_iface)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);
    unsigned max_events    = iface->config.max_poll;
    unsigned count         = 0;
    unsigned read_events;
    ucs_status_t status;

    do {
        read_events = ucs_min(ucs_sys_event_set_max_wait_events, max_events);
        status = ucs_event_set_wait(iface->event_set, &read_events,
                                    0, uct_tcp_iface_handle_events,
                                    (void *)&count);
        max_events -= read_events;
        ucs_trace_poll("iface=%p ucs_event_set_wait() returned %d: "
                       "read events=%u, total=%u",
                       iface, status, read_events,
                       iface->config.max_poll - max_events);
    } while ((max_events > 0) && (read_events == UCT_TCP_MAX_EVENTS) &&
             ((status == UCS_OK) || (status == UCS_INPROGRESS)));

    return count;
}

static ucs_status_t uct_tcp_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                        uct_completion_t *comp)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (iface->outstanding != 0) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super);
        return UCS_INPROGRESS;
    }

    UCT_TL_IFACE_STAT_FLUSH(&iface->super);
    return UCS_OK;
}

static void uct_tcp_iface_connect_handler(int listen_fd, int events, void *arg)
{
    uct_tcp_iface_t *iface = arg;
    struct sockaddr_in peer_addr;
    socklen_t addrlen;
    ucs_status_t status;
    int fd;

    ucs_assert(listen_fd == iface->listen_fd);

    for (;;) {
        addrlen = sizeof(peer_addr);
        status  = ucs_socket_accept(iface->listen_fd, (struct sockaddr*)&peer_addr,
                                    &addrlen, &fd);
        if (status != UCS_OK) {
            if (status != UCS_ERR_NO_PROGRESS) {
                ucs_close_fd(&iface->listen_fd);
            }
            return;
        }
        ucs_assert(fd != -1);

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

    status = ucs_socket_set_buffer_size(fd, iface->sockopt.sndbuf,
                                        iface->sockopt.rcvbuf);
    if (status != UCS_OK) {
        return status;
    }

    return ucs_tcp_base_set_syn_cnt(fd, iface->config.syn_cnt);
}

static uct_iface_ops_t uct_tcp_iface_ops = {
    .ep_am_short              = uct_tcp_ep_am_short,
    .ep_am_bcopy              = uct_tcp_ep_am_bcopy,
    .ep_am_zcopy              = uct_tcp_ep_am_zcopy,
    .ep_put_zcopy             = uct_tcp_ep_put_zcopy,
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

static ucs_status_t uct_tcp_iface_server_init(uct_tcp_iface_t *iface)
{
    struct sockaddr_in bind_addr = iface->config.ifaddr;
    unsigned port_range_start    = iface->port_range.first;
    unsigned port_range_end      = iface->port_range.last;
    ucs_status_t status;
    int port, retry;

    /* retry is 1 for a range of ports or when port value is zero.
     * retry is 0 for a single value port that is not zero */
    retry = (port_range_start == 0) || (port_range_start < port_range_end);

    do {
        if (port_range_end != 0) {
            status = ucs_rand_range(port_range_start, port_range_end, &port);
            if (status != UCS_OK) {
                break;
            }
        } else {
            port = 0;   /* let the operating system choose the port */
        }

        status = ucs_sockaddr_set_port((struct sockaddr *)&bind_addr, port);
        if (status != UCS_OK) {
            break;
        }

        status = ucs_socket_server_init((struct sockaddr *)&bind_addr,
                                        sizeof(bind_addr), ucs_socket_max_conn(),
                                        retry, 0, &iface->listen_fd);
    } while (retry && (status == UCS_ERR_BUSY));

    return status;
}

static ucs_status_t uct_tcp_iface_listener_init(uct_tcp_iface_t *iface)
{
    struct sockaddr_in bind_addr = iface->config.ifaddr;
    socklen_t socklen            = sizeof(bind_addr);
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;
    int ret;

    status = uct_tcp_iface_server_init(iface);
    if (status != UCS_OK) {
        goto err;
    }

    /* Get the port which was selected for the socket */
    ret = getsockname(iface->listen_fd, (struct sockaddr *)&bind_addr, &socklen);
    if (ret < 0) {
        ucs_error("getsockname(fd=%d) failed: %m", iface->listen_fd);
        status = UCS_ERR_IO_ERROR;
        goto err_close_sock;
    }

    iface->config.ifaddr.sin_port = bind_addr.sin_port;

    /* Register event handler for incoming connections */
    status = ucs_async_set_event_handler(iface->super.worker->async->mode,
                                         iface->listen_fd,
                                         UCS_EVENT_SET_EVREAD |
                                         UCS_EVENT_SET_EVERR,
                                         uct_tcp_iface_connect_handler, iface,
                                         iface->super.worker->async);
    if (status != UCS_OK) {
        goto err_close_sock;
    }

    ucs_debug("tcp_iface %p: listening for connections (fd=%d) on %s",
              iface, iface->listen_fd, ucs_sockaddr_str((struct sockaddr *)&bind_addr,
                                                       ip_port_str, sizeof(ip_port_str)));
    return UCS_OK;

err_close_sock:
    close(iface->listen_fd);
err:
    return status;
}

static ucs_status_t uct_tcp_iface_set_port_range(uct_tcp_iface_t *iface,
                                                 uct_tcp_iface_config_t *config)
{
    ucs_range_spec_t system_port_range;
    unsigned start_range, end_range;
    ucs_status_t status;

    if ((config->port_range.first == 0) && (config->port_range.last == 0)) {
        /* using a random port */
        goto out_set_config;
    }

    /* get the system configuration for usable ports range */
    status = ucs_sockaddr_get_ip_local_port_range(&system_port_range);
    if (status != UCS_OK) {
        /* using the user's config */
        goto out_set_config;
    }

    /* find a common range between the user's ports range and the one on the system */
    start_range = ucs_max(system_port_range.first, config->port_range.first);
    end_range   = ucs_min(system_port_range.last, config->port_range.last);

    if (start_range > end_range) {
        /* there is no common range */
        ucs_error("the requested TCP port range (%d-%d) is outside of system's "
                  "configured port range (%d-%d)",
                  config->port_range.first, config->port_range.last,
                  system_port_range.first, system_port_range.last);
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    iface->port_range.first = start_range;
    iface->port_range.last  = end_range;
    ucs_debug("using TCP port range: %d-%d", iface->port_range.first, iface->port_range.last);
    return UCS_OK;

out_set_config:
    iface->port_range.first = config->port_range.first;
    iface->port_range.last  = config->port_range.last;
    ucs_debug("using TCP port range: %d-%d", iface->port_range.first, iface->port_range.last);
    return UCS_OK;
out:
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

    if (ucs_derived_of(worker, uct_priv_worker_t)->thread_mode == UCS_THREAD_MODE_MULTI) {
        ucs_error("TCP transport does not support multi-threaded worker");
        return UCS_ERR_INVALID_PARAM;
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_tcp_iface_ops, md, worker,
                              params, tl_config
                              UCS_STATS_ARG((params->field_mask &
                                             UCT_IFACE_PARAM_FIELD_STATS_ROOT) ?
                                            params->stats_root : NULL)
                              UCS_STATS_ARG(params->mode.device.dev_name));

    ucs_strncpy_zero(self->if_name, params->mode.device.dev_name,
                     sizeof(self->if_name));
    self->outstanding        = 0;
    self->config.tx_seg_size = config->tx_seg_size +
                               sizeof(uct_tcp_am_hdr_t);
    self->config.rx_seg_size = config->rx_seg_size +
                               sizeof(uct_tcp_am_hdr_t);

    if (ucs_iov_get_max() >= UCT_TCP_EP_AM_SHORTV_IOV_COUNT) {
        self->config.sendv_thresh = config->sendv_thresh;
    } else {
        /* AM Short with non-blocking vector send can't be used */
        self->config.sendv_thresh = UCS_MEMUNITS_INF;
    }

    /* Maximum IOV count allowed by user's configuration (considering TCP
     * protocol and user's AM headers that use 1st and 2nd IOVs
     * correspondingly) and system constraints */
    self->config.zcopy.max_iov    = ucs_min(config->max_iov +
                                            UCT_TCP_EP_ZCOPY_SERVICE_IOV_COUNT,
                                            ucs_iov_get_max());
    /* Use a remaining part of TX segment for AM Zcopy header */
    self->config.zcopy.hdr_offset = (sizeof(uct_tcp_ep_zcopy_tx_t) +
                                     sizeof(struct iovec) *
                                     self->config.zcopy.max_iov);
    if ((self->config.zcopy.hdr_offset > self->config.tx_seg_size) &&
        (self->config.zcopy.max_iov > UCT_TCP_EP_ZCOPY_SERVICE_IOV_COUNT)) {
        ucs_error("AM Zcopy context (%zu) must be <= TX segment size (%zu). "
                  "It can be adjusted by decreasing maximum IOV count (%zu)",
                  self->config.zcopy.hdr_offset, self->config.tx_seg_size,
                  self->config.zcopy.max_iov);
        return UCS_ERR_INVALID_PARAM;
    }

    if (config->max_conn_retries > UINT8_MAX) {
        ucs_error("unsupported value was specified (%u) for the maximal "
                  "connection retries, expected lower than %u",
                  config->max_conn_retries, UINT8_MAX);
        return UCS_ERR_INVALID_PARAM;
    }

    self->config.zcopy.max_hdr     = self->config.tx_seg_size -
                                     self->config.zcopy.hdr_offset;
    self->config.prefer_default    = config->prefer_default;
    self->config.put_enable        = config->put_enable;
    self->config.conn_nb           = config->conn_nb;
    self->config.max_poll          = config->max_poll;
    self->config.max_conn_retries  = config->max_conn_retries;
    self->config.syn_cnt           = config->syn_cnt;
    self->sockopt.nodelay          = config->sockopt_nodelay;
    self->sockopt.sndbuf           = config->sockopt.sndbuf;
    self->sockopt.rcvbuf           = config->sockopt.rcvbuf;

    status = uct_tcp_iface_set_port_range(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    ucs_list_head_init(&self->ep_list);
    ucs_conn_match_init(&self->conn_match_ctx,
                        ucs_field_sizeof(uct_tcp_ep_t, peer_addr),
                        &uct_tcp_cm_conn_match_ops);

    if (self->config.tx_seg_size > self->config.rx_seg_size) {
        ucs_error("RX segment size (%zu) must be >= TX segment size (%zu)",
                  self->config.rx_seg_size, self->config.tx_seg_size);
        return UCS_ERR_INVALID_PARAM;
    }

    status = ucs_mpool_init(&self->tx_mpool, 0, self->config.tx_seg_size,
                            0, UCS_SYS_CACHE_LINE_SIZE,
                            (config->tx_mpool.bufs_grow == 0) ?
                            32 : config->tx_mpool.bufs_grow,
                            config->tx_mpool.max_bufs,
                            &uct_tcp_mpool_ops, "uct_tcp_iface_tx_buf_mp");
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_mpool_init(&self->rx_mpool, 0, self->config.rx_seg_size * 2,
                            0, UCS_SYS_CACHE_LINE_SIZE,
                            (config->rx_mpool.bufs_grow == 0) ?
                            32 : config->rx_mpool.bufs_grow,
                            config->rx_mpool.max_bufs,
                            &uct_tcp_mpool_ops, "uct_tcp_iface_rx_buf_mp");
    if (status != UCS_OK) {
        goto err_cleanup_tx_mpool;
    }

    status = uct_tcp_netif_inaddr(self->if_name, &self->config.ifaddr,
                                  &self->config.netmask);
    if (status != UCS_OK) {
        goto err_cleanup_rx_mpool;
    }

    status = ucs_event_set_create(&self->event_set);
    if (status != UCS_OK) {
        status = UCS_ERR_IO_ERROR;
        goto err_cleanup_rx_mpool;
    }

    status = uct_tcp_iface_listener_init(self);
    if (status != UCS_OK) {
        goto err_cleanup_event_set;
    }

    return UCS_OK;

err_cleanup_event_set:
    ucs_event_set_cleanup(self->event_set);
err_cleanup_rx_mpool:
    ucs_mpool_cleanup(&self->rx_mpool, 1);
err_cleanup_tx_mpool:
    ucs_mpool_cleanup(&self->tx_mpool, 1);
err:
    return status;
}

static void uct_tcp_iface_ep_list_cleanup(uct_tcp_iface_t *iface)
{
    uct_tcp_ep_t *ep, *tmp;

    ucs_list_for_each_safe(ep, tmp, &iface->ep_list, list) {
        uct_tcp_ep_destroy_internal(&ep->super.super);
    }
}

void uct_tcp_iface_add_ep(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    UCS_ASYNC_BLOCK(iface->super.worker->async);
    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_ON_MATCH_CTX));
    ucs_list_add_tail(&iface->ep_list, &ep->list);
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}

void uct_tcp_iface_remove_ep(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    UCS_ASYNC_BLOCK(iface->super.worker->async);
    ucs_list_del(&ep->list);
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}

int uct_tcp_iface_is_self_addr(uct_tcp_iface_t *iface,
                               const struct sockaddr_in *peer_addr)
{
    ucs_status_t status;
    int cmp;

    cmp = ucs_sockaddr_cmp((const struct sockaddr*)peer_addr,
                           (const struct sockaddr*)&iface->config.ifaddr,
                           &status);
    ucs_assert(status == UCS_OK);
    return !cmp;
}

static UCS_CLASS_CLEANUP_FUNC(uct_tcp_iface_t)
{
    ucs_status_t status;

    ucs_debug("tcp_iface %p: destroying", self);

    uct_base_iface_progress_disable(&self->super.super,
                                    UCT_PROGRESS_SEND |
                                    UCT_PROGRESS_RECV);

    status = ucs_async_remove_handler(self->listen_fd, 1);
    if (status != UCS_OK) {
        ucs_warn("failed to remove handler for server socket fd=%d", self->listen_fd);
    }

    uct_tcp_iface_ep_list_cleanup(self);
    ucs_conn_match_cleanup(&self->conn_match_ctx);

    ucs_mpool_cleanup(&self->rx_mpool, 1);
    ucs_mpool_cleanup(&self->tx_mpool, 1);

    ucs_close_fd(&self->listen_fd);
    ucs_event_set_cleanup(self->event_set);
}

UCS_CLASS_DEFINE(uct_tcp_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_tcp_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

ucs_status_t uct_tcp_query_devices(uct_md_h md,
                                   uct_tl_device_resource_t **devices_p,
                                   unsigned *num_devices_p)
{
    uct_tl_device_resource_t *devices, *tmp;
    static const char *netdev_dir = "/sys/class/net";
    struct dirent *entry;
    unsigned num_devices;
    ucs_status_t status;
    DIR *dir;

    dir = opendir(netdev_dir);
    if (dir == NULL) {
        ucs_error("opendir(%s) failed: %m", netdev_dir);
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    devices     = NULL;
    num_devices = 0;
    for (;;) {
        errno = 0;
        entry = readdir(dir);
        if (entry == NULL) {
            if (errno != 0) {
                ucs_error("readdir(%s) failed: %m", netdev_dir);
                ucs_free(devices);
                status = UCS_ERR_IO_ERROR;
                goto out_closedir;
            }
            break; /* no more items */
        }

        /* According to the sysfs(5) manual page, all of entries
         * has to be a symbolic link representing one of the real
         * or virtual networking devices that are visible in the
         * network namespace of the process that is accessing the
         * directory. Let's avoid checking files that are not a
         * symbolic link, e.g. "." and ".." entries */
        if (entry->d_type != DT_LNK) {
            continue;
        }

        if (!ucs_netif_is_active(entry->d_name)) {
            continue;
        }

        tmp = ucs_realloc(devices, sizeof(*devices) * (num_devices + 1),
                          "tcp devices");
        if (tmp == NULL) {
            ucs_free(devices);
            status = UCS_ERR_NO_MEMORY;
            goto out_closedir;
        }
        devices = tmp;

        ucs_snprintf_zero(devices[num_devices].name,
                          sizeof(devices[num_devices].name),
                          "%s", entry->d_name);
        devices[num_devices].type = UCT_DEVICE_TYPE_NET;
        ++num_devices;
    }

    *num_devices_p = num_devices;
    *devices_p     = devices;
    status         = UCS_OK;

out_closedir:
    closedir(dir);
out:
    return status;
}

UCT_TL_DEFINE(&uct_tcp_component, tcp, uct_tcp_query_devices, uct_tcp_iface_t,
              UCT_TCP_CONFIG_PREFIX, uct_tcp_iface_config_table,
              uct_tcp_iface_config_t);
