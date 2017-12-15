/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tcp.h"

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#if HAVE_IB
#include <uct/ib/base/ib_device.h> /* for ipoib header size */
#endif
#include <ucs/sys/string.h>
#include <linux/sockios.h>
#include <linux/types.h>
#include <linux/ethtool.h>
#include <linux/if_ether.h>
#include <sys/ioctl.h>
#include <net/if_arp.h>
#include <net/if.h>
#include <netdb.h>


typedef ssize_t (*uct_tcp_io_func_t)(int fd, void *data, size_t size, int flags);


ucs_status_t uct_tcp_socket_connect(int fd, const struct sockaddr_in *dest_addr)
{
    int ret = connect(fd, (struct sockaddr*)dest_addr, sizeof(*dest_addr));
    if (ret < 0) {
        ucs_error("connect() failed: %m"); // TODO print address
        return UCS_ERR_UNREACHABLE;
    }
    return UCS_OK;
}

ucs_status_t uct_tcp_netif_caps(const char *if_name, double *latency_p,
                                double *bandwidth_p)
{
    struct ethtool_cmd edata;
    uint32_t speed_mbps;
    ucs_status_t status;
    struct ifreq ifr;
    size_t mtu, ll_headers;
    short ether_type;

    memset(&ifr, 0, sizeof(ifr));

    edata.cmd    = ETHTOOL_GSET;
    ifr.ifr_data = (void*)&edata;
    status = ucs_netif_ioctl(if_name, SIOCETHTOOL, &ifr);
    if (status == UCS_OK) {
#if HAVE_DECL_ETHTOOL_CMD_SPEED
        speed_mbps = ethtool_cmd_speed(&edata);
#else
        speed_mbps = edata.speed;
#endif
#if HAVE_DECL_SPEED_UNKNOWN
        if (speed_mbps == SPEED_UNKNOWN) {
#else
        if ((speed_mbps == 0) || ((uint16_t)speed_mbps == (uint16_t)-1)) {
#endif
            ucs_error("speed of %s is UNKNOWN", if_name);
            return UCS_ERR_NO_DEVICE;
        }
    } else {
        speed_mbps = 100; /* Default value if SIOCETHTOOL is not supported */
    }

    status = ucs_netif_ioctl(if_name, SIOCGIFHWADDR, &ifr);
    if (status == UCS_OK) {
        ether_type = ifr.ifr_addr.sa_family;
    } else {
        ether_type = ARPHRD_ETHER;
    }

    status = ucs_netif_ioctl(if_name, SIOCGIFMTU, &ifr);
    if (status == UCS_OK) {
        mtu = ifr.ifr_mtu;
    } else {
        mtu = 1500;
    }

    switch (ether_type) {
    case ARPHRD_ETHER:
        /* https://en.wikipedia.org/wiki/Ethernet_frame */
        ll_headers = 7 + /* preamble */
                     1 + /* start-of-frame */
                     ETH_HLEN + /* src MAC + dst MAC + ethertype */
                     ETH_FCS_LEN + /* CRC */
                     12; /* inter-packet gap */
        break;
#if HAVE_IB
    case ARPHRD_INFINIBAND:
        ll_headers = UCT_IB_LRH_LEN +
                     UCT_IB_GRH_LEN +
                     UCT_IB_BTH_LEN +
                     UCT_IB_DETH_LEN + /* UD */
                     4 + 20 +          /* IPoIB */
                     UCT_IB_ICRC_LEN +
                     UCT_IB_VCRC_LEN +
                     UCT_IB_DELIM_LEN;
        break;
#endif
    default:
        ll_headers = 0;
        break;
    }

    /* https://w3.siemens.com/mcms/industrial-communication/en/rugged-communication/Documents/AN8.pdf */
    *latency_p   = 576.0 / (speed_mbps * 1e6) + 5.2e-6;
    *bandwidth_p = (speed_mbps * 1e6) / 8 *
                   (mtu - 40) / (mtu + ll_headers); /* TCP/IP header is 40 bytes */
    return UCS_OK;
}

ucs_status_t uct_tcp_netif_inaddr(const char *if_name, struct sockaddr_in *ifaddr,
                                  struct sockaddr_in *netmask)
{
    ucs_status_t status;
    struct ifreq ifra, ifrnm;

    status = ucs_netif_ioctl(if_name, SIOCGIFADDR, &ifra);
    if (status != UCS_OK) {
        return status;
    }

    if (netmask != NULL) {
        status = ucs_netif_ioctl(if_name, SIOCGIFNETMASK, &ifrnm);
        if (status != UCS_OK) {
            return status;
        }
    }

    if ((ifra.ifr_addr.sa_family  != AF_INET) ) {
        ucs_error("%s address is not INET", if_name);
        return UCS_ERR_INVALID_ADDR;
    }

    memcpy(ifaddr,  (struct sockaddr_in*)&ifra.ifr_addr,  sizeof(*ifaddr));
    if (netmask != NULL) {
        memcpy(netmask, (struct sockaddr_in*)&ifrnm.ifr_addr, sizeof(*netmask));
    }

    return UCS_OK;
}

ucs_status_t uct_tcp_netif_is_default(const char *if_name, int *result_p)
{
    static const char *filename = "/proc/net/route";
    in_addr_t netmask;
    char name[128];
    char str[128];
    FILE *f;
    int ret;

    f = fopen(filename, "r");
    if (f == NULL) {
        ucs_debug("failed to open '%s': %m", filename);
        return UCS_ERR_IO_ERROR;
    }

    /*
    Iface  Destination  Gateway  Flags  RefCnt  Use  Metric  Mask  MTU  Window  IRTT
    */
    while (fgets(str, sizeof(str), f) != NULL) {
        ret = sscanf(str, "%s %*x %*x %*d %*d %*d %*d %x", name, &netmask);
        if ((ret == 3) && !strcmp(name, if_name) && (netmask == 0)) {
            *result_p = 1;
            break;
        }

        /* Skip rest of the line */
        while ((strchr(str, '\n') == NULL) && (fgets(str, sizeof(str), f) != NULL));
    }

    *result_p = 0;
    fclose(f);
    return UCS_OK;
}

static ucs_status_t uct_tcp_do_io(int fd, void *data, size_t *length_p,
                                  uct_tcp_io_func_t io_func, const char *name)
{
    ssize_t ret;

    ucs_assert(*length_p > 0);
    ret = io_func(fd, data, *length_p, 0);
    if (ret == 0) {
        ucs_trace("fd %d is closed", fd);
        return UCS_ERR_CANCELED; /* Connection closed */
    } else if (ret < 0) {
        if ((errno == EINTR) || (errno == EAGAIN)) {
            *length_p = 0;
            return UCS_OK;
        } else {
            ucs_error("%s(fd=%d data=%p length=%zu) failed: %m",
                      name, fd, data, *length_p);
            return UCS_ERR_IO_ERROR;
        }
    } else {
        *length_p = ret;
        return UCS_OK;
    }
}

ucs_status_t uct_tcp_send(int fd, const void *data, size_t *length_p)
{
    return uct_tcp_do_io(fd, (void*)data, length_p, (uct_tcp_io_func_t)send,
                         "send");
}

ucs_status_t uct_tcp_recv(int fd, void *data, size_t *length_p)
{
    return uct_tcp_do_io(fd, data, length_p, recv, "recv");
}
