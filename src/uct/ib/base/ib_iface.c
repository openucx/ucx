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
   "Which IB service level to use.\n",
   ucs_offsetof(uct_ib_iface_config_t, sl), UCS_CONFIG_TYPE_UINT},

  {"LID_PATH_BITS", "0-17",
   "list of IB Path bits separated by comma (a,b,c) "
   "which will be the low portion of the LID, according to the LMC in the fabric.",
   ucs_offsetof(uct_ib_iface_config_t, lid_path_bits), UCS_CONFIG_TYPE_ARRAY(path_bits_spec)},

   {"PKEY_VALUE", "0x7fff",
   "Which pkey value to use. Only the partition number (lower 15 bits) is used.",
   ucs_offsetof(uct_ib_iface_config_t, pkey_value), UCS_CONFIG_TYPE_HEX},


  {NULL}
};

static ucs_status_t uct_ib_iface_find_port(uct_ib_context_t *ibctx,
                                           const char *dev_name,
                                           uct_ib_device_t **p_dev,
                                           uint8_t *p_port_num)
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

            *p_dev      = dev;
            *p_port_num = port_num;
            return UCS_OK;
        }
    }

    /* Device not found */
    return UCS_ERR_NO_DEVICE;
}

static void uct_ib_iface_recv_desc_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh)
{
    uct_ib_iface_recv_desc_t *desc = obj;
    struct ibv_mr *mr = memh;
    desc->lkey = mr->lkey;
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
                                  iface->config.rx_payload_offset + iface->config.seg_size,
                                  iface->config.rx_hdr_offset,
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &config->rx.mp,
                                  grow,
                                  uct_ib_iface_recv_desc_init,
                                  name,
                                  mp_p);
}

void uct_ib_iface_release_am_desc(uct_iface_t *tl_iface, void *desc)
{
    uct_ib_iface_t *iface = ucs_derived_of(tl_iface, uct_ib_iface_t);
    void *ib_desc;

    ib_desc = desc - iface->config.rx_headroom_offset;
    ucs_mpool_put(ib_desc);
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

/**
 * @param rx_headroom   Headroom requested by the user.
 * @param rx_priv_len   Length of transport private data to reserve (0 if unused)
 * @param rx_hdr_len    Length of transport network header.
 */
UCS_CLASS_INIT_FUNC(uct_ib_iface_t, uct_iface_ops_t *ops, uct_worker_h worker,
                    const char *dev_name, unsigned rx_headroom, unsigned rx_priv_len,
                    unsigned rx_hdr_len, unsigned tx_cq_len,
                    uct_ib_iface_config_t *config)
{
    uct_ib_context_t *ibctx = ucs_component_get(worker->context, ib, uct_ib_context_t);
    uct_ib_device_t *dev;
    ucs_status_t status;
    uint8_t port_num, lmc;
    uint16_t pkey, pkey_partition, cfg_partition_value, tbl_index;
    unsigned i, j, first, last, range_values_count;
    int ret, step, found_pkey;

    status = uct_ib_iface_find_port(ibctx, dev_name, &dev, &port_num);
    if (status != UCS_OK) {
        ucs_error("Failed to find port %s: %s", dev_name, ucs_status_string(status));
        goto err;
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, ops, worker, &dev->super,
                              &config->super UCS_STATS_ARG(dev->stats));

    self->port_num                 = port_num;
    self->sl                       = config->sl;
    self->config.rx_payload_offset = sizeof(uct_ib_iface_recv_desc_t) +
                                     ucs_max(sizeof(uct_am_recv_desc_t) + rx_headroom,
                                             rx_priv_len + rx_hdr_len);
    self->config.rx_hdr_offset     = self->config.rx_payload_offset - rx_hdr_len;
    self->config.rx_headroom_offset= self->config.rx_payload_offset - rx_headroom;
    self->config.seg_size          = config->super.max_bcopy;

    /* get the user's pkey value and find its index in the port's pkey table */
    cfg_partition_value = config->pkey_value & UCT_IB_PKEY_MASK;
    self->pkey_index = 0;
    found_pkey = 0;
    for (tbl_index = 0; tbl_index < uct_ib_iface_port_attr(self)->pkey_tbl_len; tbl_index++) {
        /* get the pkey values from the port's pkeys table */
        if (ibv_query_pkey(dev->ibv_context, port_num, tbl_index, &pkey)) {
            ucs_error("failed to get the pkey value from %s, port %d, table_index: %d: %m",
                      dev_name, port_num, tbl_index);
        }
        /* take only the lower 15 bits for the comparison */
        pkey = ntohs(pkey);
        pkey_partition = pkey & UCT_IB_PKEY_MASK;
        if (pkey_partition == cfg_partition_value) {
            self->pkey_index = tbl_index;
            found_pkey = 1;
            ucs_debug("using pkey: 0x%x. partition value: 0x%x. on %s, port %d, table_index: %d",
                      pkey, pkey_partition, dev_name, port_num, tbl_index);
            break;
        }
    }

    if ((!found_pkey) || (pkey == 0) || ((pkey & ~UCT_IB_PKEY_MASK) == 0)) {
        ucs_error("The requested pkey: 0x%x, cannot be used. "
                 "It wasn't found or the configured pkey doesn't have full membership.",
                 config->pkey_value);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    ret = ibv_query_gid(dev->ibv_context, port_num, config->gid_index, &self->gid);
    if (ret != 0) {
        ucs_error("ibv_query_gid(index=%d) failed: %m", config->gid_index);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }
    if ((self->gid.global.interface_id == 0) && (self->gid.global.subnet_prefix == 0)) {
        ucs_error("Invalid gid[%d] on %s:%d", config->gid_index,
                  uct_ib_device_name(dev), port_num);
        status = UCS_ERR_INVALID_ADDR;
        goto err;
    }

    if (config->lid_path_bits.count == 0) {
        ucs_error("List of path bits must not be empty");
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    /* count the number of lid_path_bits */
    range_values_count = 0;
    for (i = 0; i < config->lid_path_bits.count; i++) {
        range_values_count += 1 + abs(config->lid_path_bits.ranges[i].first - config->lid_path_bits.ranges[i].last);
    }

    self->path_bits = ucs_malloc(range_values_count * sizeof(*self->path_bits), "ib_path_bits");
    if (self->path_bits == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    lmc = uct_ib_iface_port_attr(self)->lmc;
    self->path_bits_count = 0;
    /* go over the list of values (ranges) for the lid_path_bits and set them */
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
                ucs_debug("Invalid value for path_bits: %d. must be < 2^lmc (lmc=%d)",
                          j, lmc);
                if (step == 1) {
                    break;
                } else {
                    continue;
                }
            }
            self->path_bits[self->path_bits_count] = j;
            self->path_bits_count++;
        }
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

int uct_ib_iface_prepare_rx_wrs(uct_ib_iface_t *iface,
                                ucs_mpool_h rx_mp, 
                                uct_ib_recv_wr_t *wrs, unsigned n)
{
    uct_ib_iface_recv_desc_t *desc;
    unsigned count;

    count = 0;
    while (count < n) {
        UCT_TL_IFACE_GET_RX_DESC(&iface->super, rx_mp, desc, break);
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

