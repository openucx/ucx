/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tcp.h"

#include <linux/sockios.h>
#include <sys/ioctl.h>
#include <net/if_arp.h>
#include <net/if.h>


ucs_status_t uct_tcp_socket_create(int *fd_p)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        ucs_error("socket create failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    *fd_p = fd;
    return UCS_OK;
}

ucs_status_t uct_tcp_socket_connect(int fd, const struct sockaddr_in *dest_addr)
{
    int ret = connect(fd, (struct sockaddr*)dest_addr, sizeof(*dest_addr));
    if (ret < 0) {
        ucs_error("connect() failed: %m"); // TODO print address
        return UCS_ERR_UNREACHABLE;
    }
    return UCS_OK;
}

static ucs_status_t uct_tcp_netif_ioctl(const char *if_name, unsigned long request,
                                        struct ifreq *if_req)
{
    ucs_status_t status;
    int fd, ret;

    strncpy(if_req->ifr_name, if_name, sizeof(if_req->ifr_name));

    status = uct_tcp_socket_create(&fd);
    if (status != UCS_OK) {
        goto out;
    }

    ret = ioctl(fd, request, if_req);
    if (ret < 0) {
        ucs_error("ioctl(req=%lu, ifr_name=%s) failed: %m", request, if_name);
        status = UCS_ERR_IO_ERROR;
        goto out_close_fd;
    }

    status = UCS_OK;

out_close_fd:
    close(fd);
out:
    return status;
}

int uct_tcp_netif_check(const char *if_name)
{
    ucs_status_t status;
    struct ifreq ifr;

    status = uct_tcp_netif_ioctl(if_name, SIOCGIFFLAGS, &ifr);
    if (status != UCS_OK) {
        return 0;
    }

    return (ifr.ifr_flags & IFF_UP) && (ifr.ifr_flags & IFF_RUNNING) &&
           !(ifr.ifr_flags & IFF_LOOPBACK);
}

ucs_status_t uct_tcp_netif_inaddr(const char *if_name, struct sockaddr_in *ifaddr)
{
    ucs_status_t status;
    struct ifreq ifra;

    status = uct_tcp_netif_ioctl(if_name, SIOCGIFADDR, &ifra);
    if (status != UCS_OK) {
        return status;
    }

    if ((ifra.ifr_addr.sa_family  != AF_INET) ) {
        ucs_error("%s address is not INET", if_name);
        return UCS_ERR_INVALID_ADDR;
    }

    memcpy(ifaddr,  (struct sockaddr_in*)&ifra.ifr_addr,  sizeof(*ifaddr));
    return UCS_OK;
}
