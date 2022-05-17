/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "tcp.h"

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


ucs_status_t uct_tcp_netif_caps(const char *if_name, double *latency_p,
                                double *bandwidth_p)
{
    struct ethtool_cmd edata;
    uint32_t speed_mbps;
    ucs_status_t status;
    struct ifreq ifr;
    size_t ll_headers;
    int speed_known;
    short ether_type;
    size_t mtu;

    memset(&ifr, 0, sizeof(ifr));

    speed_known  = 0;
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
        speed_known = speed_mbps != (uint32_t)SPEED_UNKNOWN;
#else
        speed_known = (speed_mbps != 0) && ((uint16_t)speed_mbps != (uint16_t)-1);
#endif
    }

    if (!speed_known) {
        speed_mbps = 100;
        ucs_debug("speed of %s is UNKNOWN, assuming %d Mbps", if_name, speed_mbps);
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
    case ARPHRD_INFINIBAND:
        ll_headers = /* LRH */   8  +
                     /* GRH */   40 +
                     /* BTH */   12 +
                     /* DETH */  8  +
                     /* IPoIB */ 4 + 20 +
                     /* ICRC */  4  +
                     /* VCRC */  2  +
                     /* DELIM */ 2;
        break;
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
        if ((ret == 2) && !strcmp(name, if_name) && (netmask == 0)) {
            *result_p = 1;
            goto out;
        }

        /* Skip rest of the line */
        while ((strchr(str, '\n') == NULL) && (fgets(str, sizeof(str), f) != NULL));
    }

    *result_p = 0;
out:
    fclose(f);
    return UCS_OK;
}
