/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ib_iface.h"
#include "ib_context.h"

#include <uct/tl/context.h>
#include <ucs/type/component.h>
#include <ucs/type/class.h>
#include <ucs/debug/log.h>
#include <string.h>
#include <stdlib.h>


ucs_config_field_t uct_ib_iface_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_ib_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

  {"TX_QUEUE_LEN", "256",
   "Length of send queue in the QP.",
   ucs_offsetof(uct_ib_iface_config_t, tx.queue_len), UCS_CONFIG_TYPE_UINT},

  {"TX_MAX_BATCH", "16",
   "Number of send WQEs to batch in one post-send list. Larger values reduce\n"
   "the CPU usage, but increase the latency and pipelining between sender and\n"
   "receiver.",
   ucs_offsetof(uct_ib_iface_config_t, tx.max_batch), UCS_CONFIG_TYPE_UINT},

  {"TX_MIN_INLINE", "64",
   "Bytes to reserve in send WQE for inline data. Messages which are small\n"
   "enough will be sent inline.",
   ucs_offsetof(uct_ib_iface_config_t, tx.min_inline), UCS_CONFIG_TYPE_MEMUNITS},

  {"TX_MIN_SGE", "3",
   "Number of SG entries to reserve in the send WQE.",
   ucs_offsetof(uct_ib_iface_config_t, tx.min_sge), UCS_CONFIG_TYPE_UINT},

  {"TX_CQ_MODERATION", "64",
   "Number of send WQEs for which completion is requested.",
   ucs_offsetof(uct_ib_iface_config_t, tx.cq_moderation), UCS_CONFIG_TYPE_UINT},

  UCT_IFACE_MPOOL_CONFIG_FIELDS("TX_", -1, 1024, "send",
                                ucs_offsetof(uct_ib_iface_config_t, tx.mp),
      "\nAttention: Setting this param with value != -1 is a dangerous thing\n"
      "in RC/DC and could cause deadlock or performance degradation."),

  {"RX_QUEUE_LEN", "4096",
   "Length of receive queue in the QPs.",
   ucs_offsetof(uct_ib_iface_config_t, rx.queue_len), UCS_CONFIG_TYPE_UINT},

  {"RX_MAX_BATCH", "64",
   "How many post-receives to perform in one batch.",
   ucs_offsetof(uct_ib_iface_config_t, rx.max_batch), UCS_CONFIG_TYPE_UINT},

  {"RX_INLINE", "0",
   "Number of bytes to request for inline receive. If the maximal supported size\n"
   "is smaller, it will be used instead. If it is possible to support a larger\n"
   "size than requested with the same hardware resources, it will be used instead.",
   ucs_offsetof(uct_ib_iface_config_t, rx.inl), UCS_CONFIG_TYPE_MEMUNITS},

  UCT_IFACE_MPOOL_CONFIG_FIELDS("RX_", -1, 0, "receive",
                                ucs_offsetof(uct_ib_iface_config_t, rx.mp), ""),

  {"GID_INDEX", "0",
   "Port GID index to use for RoCE.",
   ucs_offsetof(uct_ib_iface_config_t, gid_index), UCS_CONFIG_TYPE_UINT},

  {"SL", "0",
   "While IB service level to use.\n",
   ucs_offsetof(uct_ib_iface_config_t, sl), UCS_CONFIG_TYPE_UINT},

  {NULL}
};

static ucs_status_t uct_ib_iface_find_port(uct_ib_context_t *ibctx,
                                           uct_ib_iface_t *iface,
                                           const char *dev_name)
{
    uct_ib_device_t *dev;
    const char *ibdev_name;
    unsigned port_num;
    unsigned dev_index;
    size_t devname_len;
    char *p;

    p = strrchr(dev_name, ':');
    if (p == NULL) {
        return UCS_ERR_INVALID_PARAM; /* Wrong dev_name format */
    }
    devname_len = p - dev_name;

    for (dev_index = 0; dev_index < ibctx->num_devices; ++dev_index) {
        dev = ibctx->devices[dev_index];
        ibdev_name = uct_ib_device_name(dev);
        if ((strlen(ibdev_name) == devname_len) &&
            !strncmp(ibdev_name, dev_name, devname_len))
        {
            port_num = strtod(p + 1, &p);
            if (*p != '\0') {
                return UCS_ERR_INVALID_PARAM; /* Failed to parse port number */
            }
            if ((port_num < dev->first_port) || (port_num >= dev->first_port + dev->num_ports)) {
                return UCS_ERR_NO_DEVICE; /* Port number out of range */
            }

            iface->super.super.pd = &dev->super;
            iface->port_num       = port_num;
            return UCS_OK;
        }
    }

    /* Device not found */
    return UCS_ERR_NO_DEVICE;
}

static void uct_ib_iface_recv_desc_init(uct_iface_h tl_iface, void *obj, uct_lkey_t lkey)
{
    uct_ib_iface_recv_desc_t *desc = obj;
    desc->lkey = uct_ib_lkey_mr(lkey)->lkey;
}

ucs_status_t uct_ib_iface_recv_mpool_create(uct_ib_iface_t *iface,
                                            uct_ib_iface_config_t *config,
                                            const char *name, ucs_mpool_h *mp_p)
{
    unsigned grow;

    if (config->rx.queue_len < 1024) {
        grow = 1024;
    } else {
        /* We want to have some free (+10%) elements to avoid mem pool expansion */
        grow = ucs_min( (int)(1.1 * config->rx.queue_len + 0.5),
                        config->rx.mp.max_bufs);
    }

    return uct_iface_mpool_create(&iface->super.super,
                                  iface->config.rx_payload_offset + iface->config.rx_data_size,
                                  iface->config.rx_hdr_offset,
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &config->rx.mp,
                                  grow,
                                  uct_ib_iface_recv_desc_init,
                                  name,
                                  mp_p);
}

/**
 * @param rx_headroom   Headroom requested by the user.
 * @param rx_priv_len   Length of transport private data to reserve (0 if unused)
 * @param rx_hdr_len    Length of transport network header.
 */
static UCS_CLASS_INIT_FUNC(uct_ib_iface_t, uct_iface_ops_t *ops,
                           uct_context_h context, const char *dev_name,
                           unsigned rx_headroom, unsigned rx_priv_len,
                           unsigned rx_hdr_len, uct_ib_iface_config_t *config)
{
    uct_ib_context_t *ibctx = ucs_component_get(context, ib, uct_ib_context_t);
    uct_ib_device_t *dev;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(ops);

    status = uct_ib_iface_find_port(ibctx, self, dev_name);
    if (status != UCS_OK) {
        ucs_error("Failed to find port %s: %s", dev_name, ucs_status_string(status));
        goto err;
    }

    dev = uct_ib_iface_device(self);
    self->gid_index                = config->gid_index;
    self->sl                       = config->sl;
    self->config.rx_headroom       = rx_headroom;
    self->config.rx_payload_offset = sizeof(uct_ib_iface_recv_desc_t) +
                                     ucs_max(rx_headroom, rx_priv_len + rx_hdr_len);
    self->config.rx_hdr_offset     = self->config.rx_payload_offset - rx_hdr_len;
    self->config.rx_data_size      = config->super.max_bcopy;

    ucs_debug("rx_headroom=%d payload_ofs=%d hdr_ofs=%d data_sz=%d",
              self->config.rx_headroom, self->config.rx_payload_offset,
              self->config.rx_hdr_offset, self->config.rx_data_size);

    /* TODO comp_channel */
    /* TODO inline scatter for SQ */
    self->send_cq = ibv_create_cq(dev->ibv_context, config->tx.queue_len,
                                  NULL, NULL, 0);
    if (self->send_cq == NULL) {
        ucs_error("Failed to create send cq: %m");
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    /* TODO need exp API to set this value */
    if (config->rx.inl > 32 /*UCT_IB_MLX5_CQE64_MAX_INL*/) {
        ibv_exp_setenv(dev->ibv_context, "MLX5_CQE_SIZE", "128", 1);
    }

    self->recv_cq = ibv_create_cq(dev->ibv_context, config->rx.queue_len,
                                  NULL, NULL, 0);
    ibv_exp_setenv(dev->ibv_context, "MLX5_CQE_SIZE", "64", 1);

    if (self->recv_cq == NULL) {
        ucs_error("Failed to create recv cq: %m");
        goto err_destroy_send_cq;
    }

    if (uct_ib_device_is_port_ib(dev, self->port_num)) {
        self->addr.lid = uct_ib_device_port_attr(dev, self->port_num)->lid;
    } else {
        ucs_error("Unsupported link layer");
        goto err_destroy_recv_cq;
    }

    return UCS_OK;

err_destroy_recv_cq:
    ibv_destroy_cq(self->recv_cq);
err_destroy_send_cq:
    ibv_destroy_cq(self->send_cq);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ib_iface_t)
{
    int ret;

    ret = ibv_destroy_cq(self->recv_cq);
    if (ret != 0) {
        ucs_warn("ibv_destroy_cq(recv_cq) returned %d: %m", ret);
    }

    ret = ibv_destroy_cq(self->send_cq);
    if (ret != 0) {
        ucs_warn("ibv_destroy_cq(send_cq) returned %d: %m", ret);
    }
}

UCS_CLASS_DEFINE(uct_ib_iface_t, uct_base_iface_t);

int uct_ib_iface_prepare_rx_wrs(uct_ib_iface_t *iface,
                                ucs_mpool_h rx_mp, 
                                uct_ib_recv_wr_t *wrs, unsigned n)
{
    uct_ib_iface_recv_desc_t *desc;
    unsigned count;

    count = 0;
    while (count < n) {
        desc = ucs_mpool_get(rx_mp);
        if (desc == NULL) {
            break;
        }

        wrs[count].sg.addr   = (uintptr_t)uct_ib_iface_recv_desc_hdr(iface, desc);
        wrs[count].sg.length = iface->config.rx_data_size;
        wrs[count].sg.lkey   = desc->lkey;
        wrs[count].ibwr.num_sge = 1;
        wrs[count].ibwr.wr_id   = (uintptr_t)desc;
        wrs[count].ibwr.sg_list = &wrs[count].sg;  
        wrs[count].ibwr.next    = &wrs[count + 1].ibwr;
        ++count;
    }

    if (count > 0) {
        wrs[count - 1].ibwr.next = NULL;
    }

    return count;
}

