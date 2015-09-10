/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ib_iface.h"

#include <uct/tl/context.h>
#include <ucs/type/component.h>
#include <ucs/type/class.h>
#include <ucs/debug/log.h>
#include <string.h>
#include <stdlib.h>


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

  {"RX_MAX_BATCH", "16",
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

static ucs_status_t uct_ib_iface_find_port(uct_ib_device_t *dev,
                                           const char *resource_dev_name,
                                           uint8_t *p_port_num)
{
    const char *ibdev_name;
    unsigned port_num;
    size_t devname_len;
    char *p;

    p = strrchr(resource_dev_name, ':');
    if (p == NULL) {
        goto err; /* Wrong device name format */
    }
    devname_len = p - resource_dev_name;

    ibdev_name = uct_ib_device_name(dev);
    if ((strlen(ibdev_name) != devname_len) ||
        strncmp(ibdev_name, resource_dev_name, devname_len))
    {
        goto err; /* Device name is wrong */
    }

    port_num = strtod(p + 1, &p);
    if (*p != '\0') {
        goto err; /* Failed to parse port number */
    }
    if ((port_num < dev->first_port) || (port_num >= dev->first_port + dev->num_ports)) {
        goto err; /* Port number out of range */
    }

    *p_port_num = port_num;
    return UCS_OK;

err:
    return UCS_ERR_NO_DEVICE;
}

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

ucs_status_t uct_ib_iface_get_address(uct_iface_h tl_iface, struct sockaddr *addr)
{
    uct_ib_iface_t *iface = ucs_derived_of(tl_iface, uct_ib_iface_t);
    uct_sockaddr_ib_t *ib_addr = (uct_sockaddr_ib_t*)addr;

    /* TODO LMC */
    ib_addr->sib_family    = UCT_AF_INFINIBAND;
    ib_addr->lid           = uct_ib_iface_port_attr(iface)->lid;
    ib_addr->id            = 0;
    ib_addr->guid          = iface->gid.global.interface_id;
    ib_addr->subnet_prefix = iface->gid.global.subnet_prefix;
    ib_addr->qp_num        = 0;
    return UCS_OK;
}

ucs_status_t uct_ib_iface_get_subnet_address(uct_iface_h tl_iface,
                                             struct sockaddr *addr)
{
    uct_ib_iface_t *iface = ucs_derived_of(tl_iface, uct_ib_iface_t);
    uct_sockaddr_ib_subnet_t *subn_addr = (uct_sockaddr_ib_subnet_t*)addr;

    subn_addr->sib_family    = UCT_AF_INFINIBAND_SUBNET;
    subn_addr->subnet_prefix = iface->gid.global.subnet_prefix;
    return UCS_OK;
}

int uct_ib_iface_is_reachable(uct_iface_h tl_iface, const struct sockaddr *addr)
{
    uct_ib_iface_t *iface = ucs_derived_of(tl_iface, uct_ib_iface_t);

    if (addr->sa_family == UCT_AF_INFINIBAND) {
        return iface->gid.global.subnet_prefix ==
                        ((const uct_sockaddr_ib_t*)addr)->subnet_prefix;
    }

    if (addr->sa_family == UCT_AF_INFINIBAND_SUBNET) {
        return iface->gid.global.subnet_prefix ==
                        ((const uct_sockaddr_ib_subnet_t*)addr)->subnet_prefix;
    }

    return 0;
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
 */
UCS_CLASS_INIT_FUNC(uct_ib_iface_t, uct_iface_ops_t *ops, uct_pd_h pd,
                    uct_worker_h worker, const char *dev_name, unsigned rx_headroom,
                    unsigned rx_priv_len, unsigned rx_hdr_len, unsigned tx_cq_len,
                    uct_ib_iface_config_t *config)
{
    uct_ib_device_t *dev = ucs_derived_of(pd, uct_ib_device_t);
    ucs_status_t status;
    uint8_t port_num;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, ops, pd, worker,
                              &config->super UCS_STATS_ARG(dev->stats));

    status = uct_ib_iface_find_port(dev, dev_name, &port_num);
    if (status != UCS_OK) {
        ucs_error("Failed to find port %s: %s", dev_name, ucs_status_string(status));
        goto err;
    }

    self->port_num                 = port_num;
    self->sl                       = config->sl;
    self->config.rx_payload_offset = sizeof(uct_ib_iface_recv_desc_t) +
                                     ucs_max(sizeof(uct_am_recv_desc_t) + rx_headroom,
                                             rx_priv_len + rx_hdr_len);
    self->config.rx_hdr_offset     = self->config.rx_payload_offset - rx_hdr_len;
    self->config.rx_headroom_offset= self->config.rx_payload_offset - rx_headroom;
    self->config.seg_size          = config->super.max_bcopy;

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

    /* TODO comp_channel */
    /* TODO inline scatter for send SQ */
    self->send_cq = ibv_create_cq(dev->ibv_context, tx_cq_len, NULL, NULL, 0);
    if (self->send_cq == NULL) {
        ucs_error("Failed to create send cq: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_free_path_bits;
    }

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

    if (!uct_ib_device_is_port_ib(dev, self->port_num)) {
        ucs_error("Unsupported link layer");
        goto err_destroy_recv_cq;
    }

    ucs_debug("created uct_ib_iface_t headroom_ofs %d payload_ofs %d hdr_ofs %d data_sz %d",
              self->config.rx_headroom_offset, self->config.rx_payload_offset,
              self->config.rx_hdr_offset, self->config.seg_size);

    return UCS_OK;

err_destroy_recv_cq:
    ibv_destroy_cq(self->recv_cq);
err_destroy_send_cq:
    ibv_destroy_cq(self->send_cq);
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
        wrs[count].sg.length = iface->config.seg_size;
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

struct ibv_ah *uct_ib_create_ah(uct_ib_iface_t *iface, uint16_t dlid)
{
    struct ibv_ah_attr ah_attr;
    uct_ib_device_t *dev = uct_ib_iface_device(iface);

    memset(&ah_attr, 0, sizeof(ah_attr));
    ah_attr.port_num = iface->port_num;
    ah_attr.sl = iface->sl; 
    ah_attr.is_global = 0;
    ah_attr.dlid = dlid;

    return ibv_create_ah(dev->pd, &ah_attr);
}
