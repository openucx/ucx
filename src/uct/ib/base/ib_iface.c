/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ib_iface.h"

#include <uct/base/uct_md.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/cpu.h>
#include <ucs/type/component.h>
#include <ucs/type/class.h>
#include <ucs/debug/log.h>
#include <string.h>
#include <stdlib.h>
#include <poll.h>


static UCS_CONFIG_DEFINE_ARRAY(path_bits_spec,
                               sizeof(ucs_range_spec_t),
                               UCS_CONFIG_TYPE_RANGE_SPEC);

const char *uct_ib_mtu_values[] = {
    [UCT_IB_MTU_DEFAULT]    = "default",
    [UCT_IB_MTU_512]        = "512",
    [UCT_IB_MTU_1024]       = "1024",
    [UCT_IB_MTU_2048]       = "2048",
    [UCT_IB_MTU_4096]       = "4096",
    [UCT_IB_MTU_LAST]       = NULL
};


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

  {"TX_MAX_POLL", "16",
   "Max number of receive completions to pick during TX poll",
   ucs_offsetof(uct_ib_iface_config_t, tx.max_poll), UCS_CONFIG_TYPE_UINT},

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

  UCT_IFACE_MPOOL_CONFIG_FIELDS("TX_", 65536, 1024, "send",
                                ucs_offsetof(uct_ib_iface_config_t, tx.mp),
      "\nAttention: Setting this param with value != -1 is a dangerous thing\n"
      "in RC/DC and could cause deadlock or performance degradation."),

  {"RX_QUEUE_LEN", "4096",
   "Length of receive queue in the QPs.",
   ucs_offsetof(uct_ib_iface_config_t, rx.queue_len), UCS_CONFIG_TYPE_UINT},

  {"RX_MAX_BATCH", "16",
   "How many post-receives to perform in one batch.",
   ucs_offsetof(uct_ib_iface_config_t, rx.max_batch), UCS_CONFIG_TYPE_UINT},

  {"RX_MAX_POLL", "16",
   "Max number of receive completions to pick during RX poll",
   ucs_offsetof(uct_ib_iface_config_t, rx.max_poll), UCS_CONFIG_TYPE_UINT},

  {"RX_INLINE", "0",
   "Number of bytes to request for inline receive. If the maximal supported size\n"
   "is smaller, it will be used instead. If it is possible to support a larger\n"
   "size than requested with the same hardware resources, it will be used instead.",
   ucs_offsetof(uct_ib_iface_config_t, rx.inl), UCS_CONFIG_TYPE_MEMUNITS},

  UCT_IFACE_MPOOL_CONFIG_FIELDS("RX_", 65536, 0, "receive",
                                ucs_offsetof(uct_ib_iface_config_t, rx.mp), ""),

  {"GID_INDEX", "0",
   "Port GID index to use.",
   ucs_offsetof(uct_ib_iface_config_t, gid_index), UCS_CONFIG_TYPE_UINT},

  {"SL", "0",
   "Which IB service level to use.\n",
   ucs_offsetof(uct_ib_iface_config_t, sl), UCS_CONFIG_TYPE_UINT},

  {"LID_PATH_BITS", "0-17",
   "list of IB Path bits separated by comma (a,b,c) "
   "which will be the low portion of the LID, according to the LMC in the fabric.",
   ucs_offsetof(uct_ib_iface_config_t, lid_path_bits), UCS_CONFIG_TYPE_ARRAY(path_bits_spec)},

  {"PKEY", "0x7fff",
   "Which pkey value to use. Should be between 0 and 0x7fff.",
   ucs_offsetof(uct_ib_iface_config_t, pkey_value), UCS_CONFIG_TYPE_HEX},


  {NULL}
};

uct_ib_device_info_t uct_ib_device_info_table[] = {
  {4099, "ConnectX-3", 0, 10},
  {4103, "ConnectX-3 Pro", 0, 11},
  {4113, "Connect-IB", UCT_IB_DEVICE_FLAG_MLX5_PRM, 20},
  {4115, "ConnectX-4", UCT_IB_DEVICE_FLAG_MLX5_PRM, 30},
  {4117, "ConnectX-4 LX", UCT_IB_DEVICE_FLAG_MLX5_PRM, 28},
  {0, "", 0, 0}
};

static void uct_ib_iface_recv_desc_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh)
{
    uct_ib_iface_recv_desc_t *desc = obj;
    struct ibv_mr *mr = memh;
    desc->lkey = mr->lkey;
}

ucs_status_t uct_ib_iface_recv_mpool_init(uct_ib_iface_t *iface,
                                          uct_ib_iface_config_t *config,
                                          const char *name, ucs_mpool_t *mp)
{
    unsigned grow;

    if (config->rx.queue_len < 1024) {
        grow = 1024;
    } else {
        /* We want to have some free (+10%) elements to avoid mem pool expansion */
        grow = ucs_min( (int)(1.1 * config->rx.queue_len + 0.5),
                        config->rx.mp.max_bufs);
    }

    return uct_iface_mpool_init(&iface->super, mp,
                                iface->config.rx_payload_offset + iface->config.seg_size,
                                iface->config.rx_hdr_offset,
                                UCS_SYS_CACHE_LINE_SIZE,
                                &config->rx.mp, grow,
                                uct_ib_iface_recv_desc_init,
                                name);
}

void uct_ib_iface_release_am_desc(uct_iface_t *tl_iface, void *desc)
{
    uct_ib_iface_t *iface = ucs_derived_of(tl_iface, uct_ib_iface_t);
    void *ib_desc;

    ib_desc = desc - iface->config.rx_headroom_offset;
    ucs_mpool_put_inline(ib_desc);
}

ucs_status_t uct_ib_iface_get_device_address(uct_iface_h tl_iface,
                                             uct_device_addr_t *dev_addr)
{
    uct_ib_iface_t *iface = ucs_derived_of(tl_iface, uct_ib_iface_t);
    uct_ib_address_pack(uct_ib_iface_device(iface), iface->addr_type,
                        &iface->gid, uct_ib_iface_port_attr(iface)->lid,
                        (void*)dev_addr);
    return UCS_OK;
}

int uct_ib_iface_is_reachable(const uct_iface_h tl_iface, const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr)
{
    uct_ib_iface_t *iface = ucs_derived_of(tl_iface, uct_ib_iface_t);
    const uct_ib_address_t *ib_addr = (const void*)dev_addr;
    union ibv_gid gid;
    uint8_t is_global;
    uint16_t lid;

    uct_ib_address_unpack(ib_addr, &lid, &is_global, &gid);

#if HAVE_DECL_IBV_LINK_LAYER_INFINIBAND
    if (!(ib_addr->flags & UCT_IB_ADDRESS_FLAG_GID) ) {
        /* IB */
        return ((uct_ib_iface_port_attr(iface)->link_layer == IBV_LINK_LAYER_INFINIBAND) &&
                (ib_addr->flags &UCT_IB_ADDRESS_FLAG_LINK_LAYER_IB) &&
                (gid.global.subnet_prefix == iface->gid.global.subnet_prefix));
    } else {
        /* RoCE */
        /* there shouldn't be a lid and the gid flag should be on */
        return ((uct_ib_iface_port_attr(iface)->link_layer == IBV_LINK_LAYER_ETHERNET) &&
                (ib_addr->flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH) &&
                !(ib_addr->flags & UCT_IB_ADDRESS_FLAG_LID));
    }
#else
    if (!(ib_addr->flags & UCT_IB_ADDRESS_FLAG_GID) ) {
           /* IB */
           return ((ib_addr->flags &UCT_IB_ADDRESS_FLAG_LINK_LAYER_IB) &&
                   (gid.global.subnet_prefix == iface->gid.global.subnet_prefix));
       }
#endif
}

void uct_ib_iface_fill_ah_attr(uct_ib_iface_t *iface, const uct_ib_address_t *ib_addr,
                               uint8_t src_path_bits, struct ibv_ah_attr *ah_attr)
{
    memset(ah_attr, 0, sizeof(*ah_attr));

    uct_ib_address_unpack(ib_addr, &ah_attr->dlid, &ah_attr->is_global,
                          &ah_attr->grh.dgid);
    ah_attr->sl            = iface->sl;
    ah_attr->src_path_bits = src_path_bits;
    ah_attr->port_num      = iface->port_num;
}

ucs_status_t uct_ib_iface_create_ah(uct_ib_iface_t *iface,
                                    const uct_ib_address_t *ib_addr,
                                    uint8_t src_path_bits,
                                    struct ibv_ah **ah_p)
{
    struct ibv_ah_attr ah_attr;
    struct ibv_ah *ah;
    char buf[128];
    char *p, *endp;

    uct_ib_iface_fill_ah_attr(iface, ib_addr, src_path_bits, &ah_attr);
    ah = ibv_create_ah(uct_ib_iface_md(iface)->pd, &ah_attr);

    if (ah == NULL) {
        p    = buf;
        endp = buf + sizeof(buf);
        snprintf(p, endp - p, "dlid=%d sl=%d port=%d path_bits=%d",
                 ah_attr.dlid, ah_attr.sl, ah_attr.port_num, ah_attr.src_path_bits);
        p += strlen(p);

        if (ah_attr.is_global) {
            snprintf(p, endp - p, "dgid=");
            p += strlen(p);
            inet_ntop(AF_INET6, &ah_attr.grh.dgid, p, endp - p);
            p += strlen(p);
            snprintf(p, endp - p, " sgid_index=%d", ah_attr.grh.sgid_index);
        }

        ucs_error("ibv_create_ah(%s) on %s:%d failed: %m", buf,
                  uct_ib_device_name(uct_ib_iface_device(iface)), iface->port_num);
        return UCS_ERR_INVALID_ADDR;
    }

    *ah_p = ah;
    return UCS_OK;
}

static ucs_status_t uct_ib_iface_init_pkey(uct_ib_iface_t *iface,
                                           uct_ib_iface_config_t *config)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    uint16_t pkey_tbl_len = uct_ib_iface_port_attr(iface)->pkey_tbl_len;
    uint16_t pkey_index, port_pkey, pkey;

    if (config->pkey_value > UCT_IB_PKEY_PARTITION_MASK) {
        ucs_error("Requested pkey 0x%x is invalid, should be in the range 0..0x%x",
                  config->pkey_value, UCT_IB_PKEY_PARTITION_MASK);
        return UCS_ERR_INVALID_PARAM;
    }

    /* get the user's pkey value and find its index in the port's pkey table */
    for (pkey_index = 0; pkey_index < pkey_tbl_len; ++pkey_index) {
        /* get the pkey values from the port's pkeys table */
        if (ibv_query_pkey(dev->ibv_context, iface->port_num, pkey_index, &port_pkey)) {
            ucs_error("ibv_query_pkey(%s:%d, index=%d) failed: %m",
                      uct_ib_device_name(dev), iface->port_num, pkey_index);
        }

        pkey = ntohs(port_pkey);
        if (!(pkey & UCT_IB_PKEY_MEMBERSHIP_MASK)) {
            ucs_debug("skipping send-only pkey[%d]=0x%x", pkey_index, pkey);
            continue;
        }

        /* take only the lower 15 bits for the comparison */
        if ((pkey & UCT_IB_PKEY_PARTITION_MASK) == config->pkey_value) {
            iface->pkey_index = pkey_index;
            iface->pkey_value = pkey;
            ucs_debug("using pkey[%d] 0x%x on %s:%d", iface->pkey_index,
                      iface->pkey_value, uct_ib_device_name(dev), iface->port_num);
            return UCS_OK;
        }
    }

    ucs_error("The requested pkey: 0x%x, cannot be used. "
              "It wasn't found or the configured pkey doesn't have full membership.",
              config->pkey_value);
    return UCS_ERR_INVALID_PARAM;
}

#if HAVE_DECL_IBV_LINK_LAYER_ETHERNET
static int uct_ib_iface_is_gid_raw_empty(uint8_t *gid_raw)
{
    int i;

    for (i = 0; i < 16; i++) {
        if (gid_raw[0] != 0) {
            return 0;
        }
    }
    return 1;
}
#endif

static ucs_status_t uct_ib_iface_init_gid(uct_ib_iface_t *iface,
                                           uct_ib_iface_config_t *config)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    int ret;

    ret = ibv_query_gid(dev->ibv_context, iface->port_num, config->gid_index,
                        &iface->gid);
    if (ret != 0) {
        ucs_error("ibv_query_gid(index=%d) failed: %m", config->gid_index);
        return UCS_ERR_INVALID_PARAM;
    }

#if HAVE_DECL_IBV_LINK_LAYER_ETHERNET
    if (uct_ib_iface_port_attr(iface)->link_layer == IBV_LINK_LAYER_ETHERNET) {
        if (uct_ib_iface_is_gid_raw_empty(iface->gid.raw)) {
            ucs_error("Invalid gid[%d] on %s:%d", config->gid_index,
                      uct_ib_device_name(dev), iface->port_num);
            return UCS_ERR_INVALID_ADDR;
        } else {
            return UCS_OK;
        }
    }
#endif

    if ((iface->gid.global.interface_id == 0) && (iface->gid.global.subnet_prefix == 0)) {
        ucs_error("Invalid gid[%d] on %s:%d", config->gid_index,
                  uct_ib_device_name(dev), iface->port_num);
        return UCS_ERR_INVALID_ADDR;
    }

    return UCS_OK;
}

static ucs_status_t uct_ib_iface_init_lmc(uct_ib_iface_t *iface,
                                          uct_ib_iface_config_t *config)
{
    unsigned i, j, num_path_bits;
    unsigned first, last;
    uint8_t lmc;
    int step;

    if (config->lid_path_bits.count == 0) {
        ucs_error("List of path bits must not be empty");
        return UCS_ERR_INVALID_PARAM;
    }

    /* count the number of lid_path_bits */
    num_path_bits = 0;
    for (i = 0; i < config->lid_path_bits.count; i++) {
        num_path_bits += 1 + abs(config->lid_path_bits.ranges[i].first -
                                 config->lid_path_bits.ranges[i].last);
    }

    iface->path_bits = ucs_malloc(num_path_bits * sizeof(*iface->path_bits),
                                  "ib_path_bits");
    if (iface->path_bits == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    lmc = uct_ib_iface_port_attr(iface)->lmc;

    /* go over the list of values (ranges) for the lid_path_bits and set them */
    iface->path_bits_count = 0;
    for (i = 0; i < config->lid_path_bits.count; ++i) {

        first = config->lid_path_bits.ranges[i].first;
        last  = config->lid_path_bits.ranges[i].last;

        /* range of values or one value */
        if (first < last) {
            step = 1;
        } else {
            step = -1;
        }

        /* fill the value/s */
        for (j = first; j != (last + step); j += step) {
            if (j >= UCS_BIT(lmc)) {
                ucs_debug("Not using value %d for path_bits - must be < 2^lmc (lmc=%d)",
                          j, lmc);
                if (step == 1) {
                    break;
                } else {
                    continue;
                }
            }

            ucs_assert(iface->path_bits_count <= num_path_bits);
            iface->path_bits[iface->path_bits_count] = j;
            iface->path_bits_count++;
        }
    }

    return UCS_OK;
}

/**
 * @param rx_headroom   Headroom requested by the user.
 * @param rx_priv_len   Length of transport private data to reserve (0 if unused)
 * @param rx_hdr_len    Length of transport network header.
 * @param mss           Maximal segment size (transport limit).
 */
UCS_CLASS_INIT_FUNC(uct_ib_iface_t, uct_ib_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const char *dev_name, unsigned rx_headroom,
                    unsigned rx_priv_len, unsigned rx_hdr_len, unsigned tx_cq_len,
                    size_t mss, uct_ib_iface_config_t *config)
{
    uct_ib_device_t *dev = &ucs_derived_of(md, uct_ib_md_t)->dev;
    ucs_status_t status;
    uint8_t port_num;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &ops->super, md, worker,
                              &config->super UCS_STATS_ARG(dev->stats));

    status = uct_ib_device_find_port(dev, dev_name, &port_num);
    if (status != UCS_OK) {
        goto err;
    }

    self->port_num                 = port_num;
    self->sl                       = config->sl;
    self->config.rx_payload_offset = sizeof(uct_ib_iface_recv_desc_t) +
                                     ucs_max(sizeof(uct_am_recv_desc_t) + rx_headroom,
                                             rx_priv_len + rx_hdr_len);
    self->config.rx_hdr_offset     = self->config.rx_payload_offset - rx_hdr_len;
    self->config.rx_headroom_offset= self->config.rx_payload_offset - rx_headroom;
    self->config.seg_size          = ucs_min(mss, config->super.max_bcopy);
    self->config.tx_max_poll       = config->tx.max_poll;
    self->config.rx_max_poll       = config->rx.max_poll;
    self->config.rx_max_batch      = ucs_min(config->rx.max_batch,
                                             config->rx.queue_len / 4);
    self->ops                      = ops;

    status = uct_ib_iface_init_pkey(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_iface_init_gid(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_iface_init_lmc(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    self->comp_channel = ibv_create_comp_channel(dev->ibv_context);
    if (self->comp_channel == NULL) {
        ucs_error("Failed to create completion channel: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_free_path_bits;
    }

    status = ucs_sys_fcntl_modfl(self->comp_channel->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_destroy_comp_channel;
    }

    /* TODO inline scatter for send SQ */
    self->send_cq = ibv_create_cq(dev->ibv_context, tx_cq_len,
                                  NULL, self->comp_channel, 0);
    if (self->send_cq == NULL) {
        ucs_error("Failed to create send cq: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_comp_channel;
    }

    if (config->rx.inl > 32 /*UCT_IB_MLX5_CQE64_MAX_INL*/) {
        ibv_exp_setenv(dev->ibv_context, "MLX5_CQE_SIZE", "128", 1);
    }

    self->recv_cq = ibv_create_cq(dev->ibv_context, config->rx.queue_len,
                                  NULL, self->comp_channel, 0);
    ibv_exp_setenv(dev->ibv_context, "MLX5_CQE_SIZE", "64", 1);

    if (self->recv_cq == NULL) {
        ucs_error("Failed to create recv cq: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_send_cq;
    }

    /* Address scope and size */
#if HAVE_DECL_IBV_LINK_LAYER_ETHERNET
    if (uct_ib_iface_port_attr(self)->link_layer == IBV_LINK_LAYER_ETHERNET) {
           self->addr_type = UCT_IB_ADDRESS_TYPE_ETH;
       } else {
           self->addr_type = uct_ib_address_scope(self->gid.global.subnet_prefix);
       }
#else
    self->addr_type = uct_ib_address_scope(self->gid.global.subnet_prefix);
#endif

    self->addr_size  = uct_ib_address_size(self->addr_type);

    ucs_debug("created uct_ib_iface_t headroom_ofs %d payload_ofs %d hdr_ofs %d data_sz %d",
              self->config.rx_headroom_offset, self->config.rx_payload_offset,
              self->config.rx_hdr_offset, self->config.seg_size);

    return UCS_OK;

err_destroy_send_cq:
    ibv_destroy_cq(self->send_cq);
err_destroy_comp_channel:
    ibv_destroy_comp_channel(self->comp_channel);
err_free_path_bits:
    ucs_free(self->path_bits);
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

    ret = ibv_destroy_comp_channel(self->comp_channel);
    if (ret != 0) {
        ucs_warn("ibv_destroy_comp_channel(comp_channel) returned %d: %m", ret);
    }

    ucs_free(self->path_bits);
}

UCS_CLASS_DEFINE(uct_ib_iface_t, uct_base_iface_t);

int uct_ib_iface_prepare_rx_wrs(uct_ib_iface_t *iface, ucs_mpool_t *mp,
                                uct_ib_recv_wr_t *wrs, unsigned n)
{
    uct_ib_iface_recv_desc_t *desc;
    unsigned count;

    count = 0;
    while (count < n) {
        UCT_TL_IFACE_GET_RX_DESC(&iface->super, mp, desc, break);
        wrs[count].sg.addr   = (uintptr_t)uct_ib_iface_recv_desc_hdr(iface, desc);
        wrs[count].sg.length = iface->config.rx_payload_offset + iface->config.seg_size;
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

ucs_status_t uct_ib_iface_query(uct_ib_iface_t *iface, size_t xport_hdr_len,
                                uct_iface_attr_t *iface_attr)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    static const unsigned ib_port_widths[] = {
        [0] = 1,
        [1] = 4,
        [2] = 8,
        [3] = 12
    };
    uint8_t active_width, active_speed, active_mtu;
    double encoding, signal_rate, wire_speed;
    size_t mtu, width, extra_pkt_len;
    int i = 0;

    active_width = uct_ib_iface_port_attr(iface)->active_width;
    active_speed = uct_ib_iface_port_attr(iface)->active_speed;
    active_mtu   = uct_ib_iface_port_attr(iface)->active_mtu;

    /* Get active width */
    if (!ucs_is_pow2(active_width) ||
        (active_width < 1) || (ucs_ilog2(active_width) > 3))
    {
        ucs_error("Invalid active_width on %s:%d: %d",
                  uct_ib_device_name(dev), iface->port_num, active_width);
        return UCS_ERR_IO_ERROR;
    }

    memset(iface_attr, 0, sizeof(*iface_attr));

    iface_attr->device_addr_len = iface->addr_size;

    switch (active_speed) {
    case 1: /* SDR */
        iface_attr->latency = 5000e-9;
        signal_rate         = 2.5e9;
        encoding            = 8.0/10.0;
        break;
    case 2: /* DDR */
        iface_attr->latency = 2500e-9;
        signal_rate         = 5.0e9;
        encoding            = 8.0/10.0;
        break;
    case 4: /* QDR */
        iface_attr->latency = 1300e-9;
        signal_rate         = 10.0e9;
        encoding            = 8.0/10.0;
        break;
    case 8: /* FDR10 */
        iface_attr->latency = 700e-9;
        signal_rate         = 10.3125e9;
        encoding            = 64.0/66.0;
        break;
    case 16: /* FDR */
        iface_attr->latency = 700e-9;
        signal_rate         = 14.0625e9;
        encoding            = 64.0/66.0;
        break;
    case 32: /* EDR */
        iface_attr->latency = 600e-9;
        signal_rate         = 25.78125e9;
        encoding            = 64.0/66.0;
        break;
    default:
        ucs_error("Invalid active_speed on %s:%d: %d",
                  uct_ib_device_name(dev), iface->port_num, active_speed);
        return UCS_ERR_IO_ERROR;
    }

    /* Wire speed calculation: Width * SignalRate * Encoding */
    width                 = ib_port_widths[ucs_ilog2(active_width)];
    wire_speed            = (width * signal_rate * encoding) / 8.0;

    /* Calculate packet overhead  */
    mtu                   = ucs_min(uct_ib_mtu_value(active_mtu),
                                    iface->config.seg_size);

    extra_pkt_len = UCT_IB_BTH_LEN + xport_hdr_len +  UCT_IB_ICRC_LEN + UCT_IB_VCRC_LEN + UCT_IB_DELIM_LEN;

#if HAVE_DECL_IBV_LINK_LAYER_ETHERNET
    if (uct_ib_iface_port_attr(iface)->link_layer == IBV_LINK_LAYER_ETHERNET) {
        extra_pkt_len += UCT_IB_GRH_LEN + UCT_IB_ROCE_LEN;
            /* TODO check if UCT_IB_DELIM_LEN is present in RoCE as well */
    } else {
                extra_pkt_len += UCT_IB_LRH_LEN;
    }
#else
    extra_pkt_len += UCT_IB_LRH_LEN;
#endif
    iface_attr->bandwidth = (wire_speed * mtu) / (mtu + extra_pkt_len);

    /* Set priority of current device */
    iface_attr->priority = 0;
    while (uct_ib_device_info_table[i].vendor_part_id != 0) {
        if (uct_ib_device_info_table[i].vendor_part_id == dev->dev_attr.vendor_part_id) {
            iface_attr->priority = uct_ib_device_info_table[i].priority;
            break;
        }
        i++;
    }

    return UCS_OK;
}

ucs_status_t uct_ib_iface_wakeup_arm(uct_wakeup_h wakeup)
{
    int res, ack_count = 0;
    ucs_status_t status;
    struct ibv_cq *cq;
    void *cq_context;
    uct_ib_iface_t *iface = ucs_derived_of(wakeup->iface, uct_ib_iface_t);

    do {
        res = ibv_get_cq_event(iface->comp_channel, &cq, &cq_context);
        ack_count++;
    } while (res == 0);

    if (errno != EAGAIN) {
        return UCS_ERR_IO_ERROR;
    }

    if (ack_count > 1) {
        ibv_ack_cq_events(cq, ack_count - 1);
    }

    if (wakeup->events & UCT_WAKEUP_TX_COMPLETION) {
        status = iface->ops->arm_tx_cq(iface);
        if (status != UCS_OK) {
            return status;
        }
    }

    if (wakeup->events & (UCT_WAKEUP_RX_AM | UCT_WAKEUP_RX_SIGNALED_AM)) {
        status = iface->ops->arm_rx_cq(iface, 0);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

ucs_status_t uct_ib_iface_wakeup_get_fd(uct_wakeup_h wakeup, int *fd_p)
{
    *fd_p = wakeup->fd;
    return UCS_OK;
}

ucs_status_t uct_ib_iface_wakeup_wait(uct_wakeup_h wakeup)
{
    int res;
    struct pollfd polled = { .fd = wakeup->fd, .events = POLLIN };

    do {
        res = poll(&polled, 1, -1);
    } while ((res == -1) && (errno == EINTR));

    if ((res != 1) || (polled.revents != POLLIN)) {
        return UCS_ERR_IO_ERROR;
    }

    return uct_ib_iface_wakeup_arm(wakeup);
}

ucs_status_t uct_ib_iface_wakeup_open(uct_iface_h iface, unsigned events,
                                      uct_wakeup_h wakeup)
{
    uct_ib_iface_t *ib_iface = ucs_derived_of(iface, uct_ib_iface_t);
    wakeup->fd = ib_iface->comp_channel->fd;
    return UCS_OK;
}

ucs_status_t uct_ib_iface_wakeup_signal(uct_wakeup_h wakeup)
{
    return UCS_ERR_UNSUPPORTED;
}

void uct_ib_iface_wakeup_close(uct_wakeup_h wakeup)
{
}

static ucs_status_t uct_ib_iface_arm_cq(uct_ib_iface_t *iface, struct ibv_cq *cq,
                                        int solicited)
{
    int ret;

    ret = ibv_req_notify_cq(cq, solicited);
    if (ret != 0) {
        uct_ib_device_t *dev = uct_ib_iface_device(iface);
        ucs_error("ibv_req_notify_cq(%s:%d, cq) failed: %m",
                  uct_ib_device_name(dev), iface->port_num);
        return UCS_ERR_IO_ERROR;
    }
    return UCS_OK;
}

ucs_status_t uct_ib_iface_arm_tx_cq(uct_ib_iface_t *iface)
{
    return uct_ib_iface_arm_cq(iface, iface->send_cq, 0);
}

ucs_status_t uct_ib_iface_arm_rx_cq(uct_ib_iface_t *iface, int solicited)
{
    return uct_ib_iface_arm_cq(iface, iface->recv_cq, solicited);
}
