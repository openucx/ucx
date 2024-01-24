/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  ALL RIGHTS RESERVED.
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
#include <float.h>

#define UCT_TCP_IFACE_NETDEV_DIR "/sys/class/net"

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

  UCT_IFACE_MPOOL_CONFIG_FIELDS("TX_", -1, 8, 128m, 1.0, "send",
                                ucs_offsetof(uct_tcp_iface_config_t, tx_mpool), ""),

  UCT_IFACE_MPOOL_CONFIG_FIELDS("RX_", -1, 8, 128m, 1.0, "receive",
                                ucs_offsetof(uct_tcp_iface_config_t, rx_mpool), ""),

  {"PORT_RANGE", "0",
   "Generate a random TCP port number from that range. A value of zero means\n"
   "let the operating system select the port number.",
   ucs_offsetof(uct_tcp_iface_config_t, port_range), UCS_CONFIG_TYPE_RANGE_SPEC},

   {"MAX_BW", "2200MBs",
    "Upper bound to TCP iface bandwidth. 'auto' means BW is unlimited.",
    ucs_offsetof(uct_tcp_iface_config_t, max_bw), UCS_CONFIG_TYPE_BW},

#ifdef UCT_TCP_EP_KEEPALIVE
  {"KEEPIDLE", UCS_PP_MAKE_STRING(UCT_TCP_EP_DEFAULT_KEEPALIVE_IDLE) "s",
   "The time the connection needs to remain idle before TCP starts sending "
   "keepalive probes. Specifying \"inf\" disables keepalive.",
   ucs_offsetof(uct_tcp_iface_config_t, keepalive.idle),
                UCS_CONFIG_TYPE_TIME_UNITS},

  {"KEEPCNT", "auto",
   "The maximum number of keepalive probes TCP should send before "
   "dropping the connection. Specifying \"inf\" disables keepalive.",
   ucs_offsetof(uct_tcp_iface_config_t, keepalive.cnt),
                UCS_CONFIG_TYPE_ULUNITS},

  {"KEEPINTVL", UCS_PP_MAKE_STRING(UCT_TCP_EP_DEFAULT_KEEPALIVE_INTVL) "s",
   "The time between individual keepalive probes. Specifying \"inf\" disables"
   " keepalive.",
   ucs_offsetof(uct_tcp_iface_config_t, keepalive.intvl),
                UCS_CONFIG_TYPE_TIME_UNITS},
#endif /* UCT_TCP_EP_KEEPALIVE */

  {NULL}
};


static UCS_CLASS_DEFINE_DELETE_FUNC(uct_tcp_iface_t, uct_iface_t);

static ucs_status_t uct_tcp_iface_get_device_address(uct_iface_h tl_iface,
                                                     uct_device_addr_t *addr)
{
    uct_tcp_iface_t *iface          = ucs_derived_of(tl_iface, uct_tcp_iface_t);
    uct_tcp_device_addr_t *dev_addr = (uct_tcp_device_addr_t*)addr;
    void *pack_ptr                   = dev_addr + 1;
    const struct sockaddr *saddr    = (struct sockaddr*)&iface->config.ifaddr;
    const void *in_addr;
    size_t ip_addr_len;
    ucs_status_t status;

    dev_addr->flags     = 0;
    dev_addr->sa_family = saddr->sa_family;

    if (ucs_sockaddr_is_inaddr_loopback(saddr)) {
        dev_addr->flags |= UCT_TCP_DEVICE_ADDR_FLAG_LOOPBACK;
        memset(pack_ptr, 0, sizeof(uct_iface_local_addr_ns_t));
        uct_iface_get_local_address(pack_ptr, UCS_SYS_NS_TYPE_NET);
    } else {
        in_addr = ucs_sockaddr_get_inet_addr(saddr);
        status  = ucs_sockaddr_inet_addr_sizeof(saddr, &ip_addr_len);
        if (status != UCS_OK) {
            return status;
        }

        memcpy(pack_ptr, in_addr, ip_addr_len);
    }

    return UCS_OK;
}

static size_t uct_tcp_iface_get_device_address_length(uct_tcp_iface_t *iface)
{
    const struct sockaddr *saddr = (struct sockaddr*)&iface->config.ifaddr;
    size_t addr_len              = sizeof(uct_tcp_device_addr_t);
    size_t in_addr_len;
    ucs_status_t status;

    if (ucs_sockaddr_is_inaddr_loopback(saddr)) {
        addr_len += sizeof(uct_iface_local_addr_ns_t);
    } else {
        status = ucs_sockaddr_inet_addr_sizeof(saddr, &in_addr_len);
        ucs_assert_always(status == UCS_OK);

        addr_len += in_addr_len;
    }

    return addr_len;
}

static ucs_status_t
uct_tcp_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *addr)
{
    uct_tcp_iface_t *iface           = ucs_derived_of(tl_iface,
                                                      uct_tcp_iface_t);
    uct_tcp_iface_addr_t *iface_addr = (uct_tcp_iface_addr_t*)addr;
    ucs_status_t status;
    uint16_t port;

    status = ucs_sockaddr_get_port((struct sockaddr*)&iface->config.ifaddr,
                                   &port);
    if (status != UCS_OK) {
        return status;
    }

    iface_addr->port = htons(port);

    return UCS_OK;
}

static int
uct_tcp_iface_is_reachable_v2(const uct_iface_h tl_iface,
                              const uct_iface_is_reachable_params_t *params)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);
    uct_iface_local_addr_ns_t *local_addr_ns;
    uct_tcp_device_addr_t *tcp_dev_addr;

    if (!uct_iface_is_reachable_params_valid(
                params, UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR)) {
        return 0;
    }

    tcp_dev_addr = (uct_tcp_device_addr_t*)params->device_addr;
    if (iface->config.ifaddr.ss_family != tcp_dev_addr->sa_family) {
        return 0;
    }

    /* Loopback can connect only to loopback */
    if (!!(tcp_dev_addr->flags & UCT_TCP_DEVICE_ADDR_FLAG_LOOPBACK) !=
        ucs_sockaddr_is_inaddr_loopback(
                (const struct sockaddr*)&iface->config.ifaddr)) {
        return 0;
    }

    if (tcp_dev_addr->flags & UCT_TCP_DEVICE_ADDR_FLAG_LOOPBACK) {
        local_addr_ns = (uct_iface_local_addr_ns_t*)(tcp_dev_addr + 1);
        if (!uct_iface_local_is_reachable(local_addr_ns, UCS_SYS_NS_TYPE_NET)) {
            return 0;
        }
    }

    /* Later connect() call can still fail if the peer is actually unreachable
     * at UCT/TCP EP creation time */
    return uct_iface_scope_is_reachable(tl_iface, params);
}

static const char *
uct_tcp_iface_get_sysfs_path(const char *dev_name, char *path_buffer)
{
    ucs_status_t status;
    const char *sysfs_path;
    char lowest_path_buf[PATH_MAX];

    /* Deep search to find the lowest device sysfs path:
     * 1) For regular device, use regular sysfs form.
     * 2) For virtual device (RoCE LAG/VLAN), search for symbolic link of the
     *    form "lower_*" */
    status = ucs_netif_get_lowest_device_path(dev_name, lowest_path_buf,
                                              PATH_MAX);
    if (status != UCS_OK) {
        return NULL;
    }

    /* 'path_buffer' size is PATH_MAX */
    sysfs_path = ucs_topo_resolve_sysfs_path(lowest_path_buf, path_buffer);
    return sysfs_path;
}

static ucs_status_t uct_tcp_iface_query(uct_iface_h tl_iface,
                                        uct_iface_attr_t *attr)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);
    size_t am_buf_size     = iface->config.tx_seg_size -
                             sizeof(uct_tcp_am_hdr_t);
    ucs_status_t status;
    int is_default;
    double pci_bw, network_bw, calculated_bw;
    char path_buffer[PATH_MAX];
    const char *sysfs_path;

    uct_base_iface_query(&iface->super, attr);

    status = uct_tcp_netif_caps(iface->if_name, &attr->latency.c, &network_bw);
    if (status != UCS_OK) {
        return status;
    }

    sysfs_path             = uct_tcp_iface_get_sysfs_path(iface->if_name, path_buffer);
    pci_bw                 = ucs_topo_get_pci_bw(iface->if_name, sysfs_path);
    calculated_bw          = ucs_min(pci_bw, network_bw);

    /* Bandwidth is bounded by TCP stack computation time */
    attr->bandwidth.shared = ucs_min(calculated_bw, iface->config.max_bw);

    attr->ep_addr_len      = sizeof(uct_tcp_ep_addr_t);
    attr->iface_addr_len   = sizeof(uct_tcp_iface_addr_t);
    attr->device_addr_len  = uct_tcp_iface_get_device_address_length(iface);
    attr->cap.flags        = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                             UCT_IFACE_FLAG_CONNECT_TO_EP    |
                             UCT_IFACE_FLAG_AM_SHORT         |
                             UCT_IFACE_FLAG_AM_BCOPY         |
                             UCT_IFACE_FLAG_PENDING          |
                             UCT_IFACE_FLAG_CB_SYNC          |
                             UCT_IFACE_FLAG_EP_CHECK         |
                             UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;
    attr->cap.event_flags  = UCT_IFACE_FLAG_EVENT_SEND_COMP |
                             UCT_IFACE_FLAG_EVENT_RECV      |
                             UCT_IFACE_FLAG_EVENT_FD;

    attr->cap.am.max_short = am_buf_size;
    attr->cap.am.max_bcopy = am_buf_size;

    if (uct_tcp_keepalive_is_enabled(iface)) {
        attr->cap.flags   |= UCT_IFACE_FLAG_EP_KEEPALIVE;
    }

    if (iface->config.max_iov > UCT_TCP_EP_ZCOPY_SERVICE_IOV_COUNT) {
        /* AM */
        attr->cap.am.max_iov          = iface->config.max_iov -
                                        UCT_TCP_EP_ZCOPY_SERVICE_IOV_COUNT;
        attr->cap.am.max_zcopy        = iface->config.rx_seg_size -
                                        sizeof(uct_tcp_am_hdr_t);
        attr->cap.am.max_hdr          = iface->config.zcopy.max_hdr;
        attr->cap.am.opt_zcopy_align  = 1;
        attr->cap.flags              |= UCT_IFACE_FLAG_AM_ZCOPY;

        if (iface->config.put_enable) {
            /* PUT */
            attr->cap.put.max_iov          = iface->config.max_iov -
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
                                        ucs_event_set_types_t events,
                                        void *arg)
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

static void
uct_tcp_iface_connect_handler(int listen_fd, ucs_event_set_types_t events,
                              void *arg)
{
    uct_tcp_iface_t *iface = arg;
    struct sockaddr_storage peer_addr;
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

        status = uct_tcp_cm_handle_incoming_conn(iface,
                                                 (struct sockaddr*)&peer_addr,
                                                 fd);
        if (status != UCS_OK) {
            ucs_close_fd(&fd);
            return;
        }
    }
}

ucs_status_t uct_tcp_iface_set_sockopt(uct_tcp_iface_t *iface, int fd,
                                       int set_nb)
{
    ucs_status_t status;

    if (set_nb) {
        status = ucs_sys_fcntl_modfl(fd, O_NONBLOCK, 0);
        if (status != UCS_OK) {
            return status;
        }
    }

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
    .ep_am_short_iov          = uct_tcp_ep_am_short_iov,
    .ep_am_bcopy              = uct_tcp_ep_am_bcopy,
    .ep_am_zcopy              = uct_tcp_ep_am_zcopy,
    .ep_put_zcopy             = uct_tcp_ep_put_zcopy,
    .ep_pending_add           = uct_tcp_ep_pending_add,
    .ep_pending_purge         = uct_tcp_ep_pending_purge,
    .ep_flush                 = uct_tcp_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_check                 = uct_tcp_ep_check,
    .ep_create                = uct_tcp_ep_create,
    .ep_destroy               = uct_tcp_ep_destroy,
    .ep_get_address           = uct_tcp_ep_get_address,
    .ep_connect_to_ep         = uct_base_ep_connect_to_ep,
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
    .iface_is_reachable       = uct_base_iface_is_reachable
};

static ucs_status_t uct_tcp_iface_server_init(uct_tcp_iface_t *iface)
{
    struct sockaddr_storage bind_addr = iface->config.ifaddr;
    unsigned port_range_start         = iface->port_range.first;
    unsigned port_range_end           = iface->port_range.last;
    ucs_status_t status;
    size_t addr_len;
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

        status = ucs_sockaddr_set_port((struct sockaddr*)&bind_addr, port);
        if (status != UCS_OK) {
            break;
        }

        status = ucs_sockaddr_sizeof((struct sockaddr*)&bind_addr, &addr_len);
        if (status != UCS_OK) {
            return status;
        }

        status = ucs_socket_server_init((struct sockaddr*)&bind_addr, addr_len,
                                        ucs_socket_max_conn(), retry, 0,
                                        &iface->listen_fd);
    } while (retry && (status == UCS_ERR_BUSY));

    return status;
}

static ucs_status_t uct_tcp_iface_listener_init(uct_tcp_iface_t *iface)
{
    struct sockaddr_storage bind_addr = iface->config.ifaddr;
    socklen_t socklen                 = sizeof(bind_addr);
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;
    uint16_t port;
    int ret;

    status = uct_tcp_iface_server_init(iface);
    if (status != UCS_OK) {
        goto err;
    }

    /* Get the port which was selected for the socket */
    ret = getsockname(iface->listen_fd, (struct sockaddr*)&bind_addr, &socklen);
    if (ret < 0) {
        ucs_error("getsockname(fd=%d) failed: %m", iface->listen_fd);
        status = UCS_ERR_IO_ERROR;
        goto err_close_sock;
    }

    status = ucs_sockaddr_get_port((struct sockaddr*)&bind_addr, &port);
    if (status != UCS_OK) {
        goto err_close_sock;
    }

    status = ucs_sockaddr_set_port((struct sockaddr*)&iface->config.ifaddr,
                                   port);
    if (status != UCS_OK) {
        goto err_close_sock;
    }

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

    ucs_debug("tcp_iface %p: listening for connections (fd=%d) on %s netif %s",
              iface, iface->listen_fd,
              ucs_sockaddr_str((struct sockaddr *)&iface->config.ifaddr,
                              ip_port_str, sizeof(ip_port_str)),
              iface->if_name);
    return UCS_OK;

err_close_sock:
    ucs_close_fd(&iface->listen_fd);
err:
    return status;
}

static ucs_mpool_ops_t uct_tcp_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL,
    .obj_str       = NULL
};

static uct_iface_internal_ops_t uct_tcp_iface_internal_ops = {
    .iface_estimate_perf   = uct_base_iface_estimate_perf,
    .iface_vfs_refresh     = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
    .ep_query              = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
    .ep_invalidate         = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported,
    .ep_connect_to_ep_v2   = uct_tcp_ep_connect_to_ep_v2,
    .iface_is_reachable_v2 = uct_tcp_iface_is_reachable_v2,
    .ep_is_connected       = uct_tcp_ep_is_connected
};

static UCS_CLASS_INIT_FUNC(uct_tcp_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_tcp_iface_config_t *config = ucs_derived_of(tl_config,
                                                    uct_tcp_iface_config_t);
    uct_tcp_md_t *tcp_md           = ucs_derived_of(md, uct_tcp_md_t);
    ucs_status_t status;
    int i;
    ucs_mpool_params_t mp_params;

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

    UCS_CLASS_CALL_SUPER_INIT(
            uct_base_iface_t, &uct_tcp_iface_ops, &uct_tcp_iface_internal_ops,
            md, worker, params,
            tl_config UCS_STATS_ARG(
                    (params->field_mask & UCT_IFACE_PARAM_FIELD_STATS_ROOT) ?
                            params->stats_root :
                            NULL) UCS_STATS_ARG(params->mode.device.dev_name));

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
    self->config.max_iov          = ucs_min(config->max_iov +
                                            UCT_TCP_EP_ZCOPY_SERVICE_IOV_COUNT,
                                            ucs_iov_get_max());
    /* Use a remaining part of TX segment for AM Zcopy header */
    self->config.zcopy.hdr_offset = (sizeof(uct_tcp_ep_zcopy_tx_t) +
                                     sizeof(struct iovec) * self->config.max_iov);
    if ((self->config.zcopy.hdr_offset > self->config.tx_seg_size) &&
        (self->config.max_iov > UCT_TCP_EP_ZCOPY_SERVICE_IOV_COUNT)) {
        ucs_error("AM Zcopy context (%zu) must be <= TX segment size (%zu). "
                  "It can be adjusted by decreasing maximum IOV count (%zu)",
                  self->config.zcopy.hdr_offset, self->config.tx_seg_size,
                  self->config.max_iov);
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
    self->config.keepalive.cnt     = config->keepalive.cnt;
    self->config.keepalive.intvl   = config->keepalive.intvl;
    self->port_range.first         = config->port_range.first;
    self->port_range.last          = config->port_range.last;

    if (config->keepalive.idle != UCS_MEMUNITS_AUTO) {
        /* TCP iface configuration sets the keepalive interval */
        self->config.keepalive.idle = config->keepalive.idle;
    } else if (params->field_mask & UCT_IFACE_PARAM_FIELD_KEEPALIVE_INTERVAL) {
        /* User parameters set the keepalive interval */
        self->config.keepalive.idle = params->keepalive_interval;
    } else {
        /* Use the default keepalive interval */
        self->config.keepalive.idle =
            ucs_time_from_sec(UCT_TCP_EP_DEFAULT_KEEPALIVE_IDLE);
    }

    self->config.max_bw = UCS_CONFIG_DBL_IS_AUTO(config->max_bw) ?
                                  DBL_MAX :
                                  config->max_bw;

    if (self->config.tx_seg_size > self->config.rx_seg_size) {
        ucs_error("RX segment size (%zu) must be >= TX segment size (%zu)",
                  self->config.rx_seg_size, self->config.tx_seg_size);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    ucs_mpool_params_reset(&mp_params);
    uct_iface_mpool_config_copy(&mp_params, &config->tx_mpool);
    mp_params.elems_per_chunk = (config->tx_mpool.bufs_grow == 0) ?
                                32 : config->tx_mpool.bufs_grow;
    mp_params.elem_size       = self->config.tx_seg_size;
    mp_params.ops             = &uct_tcp_mpool_ops;
    mp_params.name            = "uct_tcp_iface_tx_buf_mp";
    status = ucs_mpool_init(&mp_params, &self->tx_mpool);
    if (status != UCS_OK) {
        goto err;
    }

    ucs_mpool_params_reset(&mp_params);
    uct_iface_mpool_config_copy(&mp_params, &config->rx_mpool);
    mp_params.elems_per_chunk = (config->rx_mpool.bufs_grow == 0) ?
                                32 : config->rx_mpool.bufs_grow;
    mp_params.elem_size       = self->config.rx_seg_size * 2;
    mp_params.ops             = &uct_tcp_mpool_ops;
    mp_params.name            = "uct_tcp_iface_rx_buf_mp";
    status = ucs_mpool_init(&mp_params, &self->rx_mpool);
    if (status != UCS_OK) {
        goto err_cleanup_tx_mpool;
    }

    for (i = 0; i < tcp_md->config.af_prio_count; i++) {
        status = ucs_netif_get_addr(self->if_name,
                                    tcp_md->config.af_prio_list[i],
                                    (struct sockaddr*)&self->config.ifaddr,
                                    (struct sockaddr*)&self->config.netmask);
        if (status == UCS_OK) {
            break;
        }
    }

    if (status != UCS_OK) {
        goto err_cleanup_rx_mpool;
    }

    status = ucs_sockaddr_sizeof((struct sockaddr*)&self->config.ifaddr,
                                 &self->config.sockaddr_len);
    if (status != UCS_OK) {
        return status;
    }

    ucs_list_head_init(&self->ep_list);
    ucs_conn_match_init(&self->conn_match_ctx, self->config.sockaddr_len,
                        UCT_TCP_CM_CONN_SN_MAX, &uct_tcp_cm_conn_match_ops);
    status = UCS_PTR_MAP_INIT(tcp_ep, &self->ep_ptr_map);
    ucs_assert_always(status == UCS_OK);

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
    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_ON_MATCH_CTX));
    ucs_list_del(&ep->list);
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}

int uct_tcp_iface_is_self_addr(uct_tcp_iface_t *iface,
                               const struct sockaddr *peer_addr)
{
    ucs_status_t status;
    int cmp;

    cmp = ucs_sockaddr_cmp(peer_addr,
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
    UCS_PTR_MAP_DESTROY(tcp_ep, &self->ep_ptr_map);

    ucs_mpool_cleanup(&self->rx_mpool, 1);
    ucs_mpool_cleanup(&self->tx_mpool, 1);

    ucs_close_fd(&self->listen_fd);
    ucs_event_set_cleanup(self->event_set);
}

UCS_CLASS_DEFINE(uct_tcp_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_tcp_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static int uct_tcp_is_bridge(const char *if_name)
{
    char path[PATH_MAX];
    struct stat st;

    ucs_snprintf_safe(path, PATH_MAX, UCT_TCP_IFACE_NETDEV_DIR "/%s/bridge",
                      if_name);

    return (stat(path, &st) == 0) && S_ISDIR(st.st_mode);
}

ucs_status_t uct_tcp_query_devices(uct_md_h md,
                                   uct_tl_device_resource_t **devices_p,
                                   unsigned *num_devices_p)
{
    uct_tcp_md_t *tcp_md               = ucs_derived_of(md, uct_tcp_md_t);
    const unsigned sys_device_priority = 10;
    uct_tl_device_resource_t *devices, *tmp;
    struct dirent **entries, **entry;
    unsigned num_devices;
    int is_active, i, n;
    ucs_status_t status;
    const char *sysfs_path;
    char path_buffer[PATH_MAX];
    ucs_sys_device_t sys_dev;

    n = scandir(UCT_TCP_IFACE_NETDEV_DIR, &entries, NULL, alphasort);
    if (n == -1) {
        ucs_error("scandir(%s) failed: %m", UCT_TCP_IFACE_NETDEV_DIR);
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    devices     = NULL;
    num_devices = 0;
    ucs_carray_for_each(entry, entries, n) {
        /* According to the sysfs(5) manual page, all of entries
         * has to be a symbolic link representing one of the real
         * or virtual networking devices that are visible in the
         * network namespace of the process that is accessing the
         * directory. Let's avoid checking files that are not a
         * symbolic link, e.g. "." and ".." entries */
        if ((*entry)->d_type != DT_LNK) {
            continue;
        }

        is_active = 0;
        for (i = 0; i < tcp_md->config.af_prio_count; i++) {
            if (ucs_netif_is_active((*entry)->d_name,
                                    tcp_md->config.af_prio_list[i])) {
                is_active = 1;
                break;
            }
        }

        if (!is_active) {
            continue;
        }

        if (!tcp_md->config.bridge_enable &&
            uct_tcp_is_bridge((*entry)->d_name)) {
            ucs_debug("filtered out bridge device %s", (*entry)->d_name);
            continue;
        }

        tmp = ucs_realloc(devices, sizeof(*devices) * (num_devices + 1),
                          "tcp devices");
        if (tmp == NULL) {
            ucs_free(devices);
            status = UCS_ERR_NO_MEMORY;
            goto out_release;
        }
        devices = tmp;

        sysfs_path = uct_tcp_iface_get_sysfs_path((*entry)->d_name,
                                                  path_buffer);
        sys_dev    = ucs_topo_get_sysfs_dev((*entry)->d_name, sysfs_path,
                                            sys_device_priority);

        ucs_snprintf_zero(devices[num_devices].name,
                          sizeof(devices[num_devices].name), "%s",
                          (*entry)->d_name);
        devices[num_devices].type       = UCT_DEVICE_TYPE_NET;
        devices[num_devices].sys_device = sys_dev;
        ++num_devices;
    }

    *num_devices_p = num_devices;
    *devices_p     = devices;
    status         = UCS_OK;

out_release:
    ucs_carray_for_each(entry, entries, n) {
        free(*entry);
    }

    free(entries);
out:
    return status;
}

int uct_tcp_keepalive_is_enabled(uct_tcp_iface_t *iface)
{
#ifdef UCT_TCP_EP_KEEPALIVE
    return (iface->config.keepalive.idle != UCS_TIME_INFINITY) &&
           (iface->config.keepalive.cnt != UCS_ULUNITS_INF) &&
           (iface->config.keepalive.intvl != UCS_TIME_INFINITY);
#else /* UCT_TCP_EP_KEEPALIVE */
    return 0;
#endif /* UCT_TCP_EP_KEEPALIVE */
}

UCT_TL_DEFINE_ENTRY(&uct_tcp_component, tcp, uct_tcp_query_devices,
                    uct_tcp_iface_t, UCT_TCP_CONFIG_PREFIX,
                    uct_tcp_iface_config_table, uct_tcp_iface_config_t);

UCT_SINGLE_TL_INIT(&uct_tcp_component, tcp,,,)
