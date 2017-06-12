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
#include <ucs/type/cpu_set.h>
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

enum {
    UCT_IB_IFACE_ADDRESS_TYPE_AUTO  = UCT_IB_ADDRESS_TYPE_LAST,
    UCT_IB_IFACE_ADDRESS_TYPE_LAST
};

static const char *uct_ib_iface_addr_types[] = {
   [UCT_IB_ADDRESS_TYPE_LINK_LOCAL] = "ib_local",
   [UCT_IB_ADDRESS_TYPE_SITE_LOCAL] = "ib_site_local",
   [UCT_IB_ADDRESS_TYPE_GLOBAL]     = "ib_global",
   [UCT_IB_ADDRESS_TYPE_ETH]        = "eth",
   [UCT_IB_IFACE_ADDRESS_TYPE_AUTO] = "auto",
   [UCT_IB_IFACE_ADDRESS_TYPE_LAST] = NULL
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

  {"TX_INLINE_RESP", "32",
   "Bytes to reserve in send WQE for inline response. Responses which are small\n"
   "enough, such as of atomic operations and small reads, will be received inline.",
   ucs_offsetof(uct_ib_iface_config_t, tx.inl_resp), UCS_CONFIG_TYPE_MEMUNITS},

  {"TX_MIN_SGE", "3",
   "Number of SG entries to reserve in the send WQE.",
   ucs_offsetof(uct_ib_iface_config_t, tx.min_sge), UCS_CONFIG_TYPE_UINT},

  {"TX_CQ_MODERATION", "64",
   "Number of send WQEs for which completion is requested.",
   ucs_offsetof(uct_ib_iface_config_t, tx.cq_moderation), UCS_CONFIG_TYPE_UINT},

  UCT_IFACE_MPOOL_CONFIG_FIELDS("TX_", -1, 1024, 1024, "send",
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

  UCT_IFACE_MPOOL_CONFIG_FIELDS("RX_", -1, 0, 0, "receive",
                                ucs_offsetof(uct_ib_iface_config_t, rx.mp), ""),

  {"ADDR_TYPE", "auto",
   "Set the interface address type. \"auto\" mode detects the type according to\n"
   "link layer type and IB subnet prefix.",
   ucs_offsetof(uct_ib_iface_config_t, addr_type),
   UCS_CONFIG_TYPE_ENUM(uct_ib_iface_addr_types)},

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

static void uct_ib_iface_recv_desc_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh)
{
    uct_ib_iface_recv_desc_t *desc = obj;
    uct_ib_mem_t *ib_memh = memh;

    desc->lkey = ib_memh->lkey;
}

ucs_status_t uct_ib_iface_recv_mpool_init(uct_ib_iface_t *iface,
                                          const uct_ib_iface_config_t *config,
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
                                &config->rx.mp, grow, grow,
                                uct_ib_iface_recv_desc_init,
                                name);
}

void uct_ib_iface_release_desc(uct_recv_desc_t *self, void *desc)
{
    uct_ib_iface_t *iface = ucs_container_of(self, uct_ib_iface_t, release_desc);
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

    switch (iface->addr_type) {
    case UCT_IB_ADDRESS_TYPE_LINK_LOCAL:
        /* IB */
        return ib_addr->flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_IB;
    case UCT_IB_ADDRESS_TYPE_SITE_LOCAL:
    case UCT_IB_ADDRESS_TYPE_GLOBAL:
        /* IB + same subnet prefix */
        return (ib_addr->flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_IB) &&
               (gid.global.subnet_prefix == iface->gid.global.subnet_prefix);
    case UCT_IB_ADDRESS_TYPE_ETH:
        /* there shouldn't be a lid and the gid flag should be on */
        return (ib_addr->flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH) &&
               (ib_addr->flags & UCT_IB_ADDRESS_FLAG_GID) &&
               !(ib_addr->flags & UCT_IB_ADDRESS_FLAG_LID);
    default:
        return 0;
    }
}

void uct_ib_iface_fill_ah_attr(uct_ib_iface_t *iface, const uct_ib_address_t *ib_addr,
                               uint8_t path_bits, struct ibv_ah_attr *ah_attr)
{
    uint8_t is_global;

    memset(ah_attr, 0, sizeof(*ah_attr));

    uct_ib_address_unpack(ib_addr, &ah_attr->dlid, &is_global, &ah_attr->grh.dgid);
    ah_attr->sl            = iface->config.sl;
    ah_attr->src_path_bits = path_bits;
    ah_attr->dlid         |= path_bits;
    ah_attr->port_num      = iface->config.port_num;

    /* Create a global address only if we cannot reach the destination using local
     * address. It means either an Ethernet address, or IB address on a different subnet.
     */
    if (is_global &&
        ((iface->addr_type == UCT_IB_ADDRESS_TYPE_ETH) ||
         (iface->gid.global.subnet_prefix != ah_attr->grh.dgid.global.subnet_prefix)))
    {
        ah_attr->is_global      = 1;
        ah_attr->grh.sgid_index = iface->config.gid_index;
    } else {
        ah_attr->is_global      = 0;
    }
}

ucs_status_t uct_ib_iface_create_ah(uct_ib_iface_t *iface,
                                    const uct_ib_address_t *ib_addr,
                                    uint8_t path_bits,
                                    struct ibv_ah **ah_p,
                                    int *is_global_p)
{
    struct ibv_ah_attr ah_attr;
    struct ibv_ah *ah;
    char buf[128];
    char *p, *endp;

    uct_ib_iface_fill_ah_attr(iface, ib_addr, path_bits, &ah_attr);
    ah = ibv_create_ah(uct_ib_iface_md(iface)->pd, &ah_attr);

    if (ah == NULL) {
        p    = buf;
        endp = buf + sizeof(buf);
        snprintf(p, endp - p, "dlid=%d sl=%d port=%d src_path_bits=%d",
                 ah_attr.dlid, ah_attr.sl, ah_attr.port_num, ah_attr.src_path_bits);
        p += strlen(p);

        if (ah_attr.is_global) {
            snprintf(p, endp - p, " dgid=");
            p += strlen(p);
            inet_ntop(AF_INET6, &ah_attr.grh.dgid, p, endp - p);
            p += strlen(p);
            snprintf(p, endp - p, " sgid_index=%d", ah_attr.grh.sgid_index);
        }

        ucs_error("ibv_create_ah(%s) on "UCT_IB_IFACE_FMT" failed: %m", buf,
                  UCT_IB_IFACE_ARG(iface));
        return UCS_ERR_INVALID_ADDR;
    }

    *ah_p        = ah;
    *is_global_p = ah_attr.is_global;
    return UCS_OK;
}

static ucs_status_t uct_ib_iface_init_pkey(uct_ib_iface_t *iface,
                                           const uct_ib_iface_config_t *config)
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
        if (ibv_query_pkey(dev->ibv_context, iface->config.port_num, pkey_index,
                           &port_pkey))
        {
            ucs_error("ibv_query_pkey("UCT_IB_IFACE_FMT", index=%d) failed: %m",
                      UCT_IB_IFACE_ARG(iface), pkey_index);
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
            ucs_debug("using pkey[%d] 0x%x on "UCT_IB_IFACE_FMT, iface->pkey_index,
                      iface->pkey_value, UCT_IB_IFACE_ARG(iface));
            return UCS_OK;
        }
    }

    ucs_error("The requested pkey: 0x%x, cannot be used. "
              "It wasn't found or the configured pkey doesn't have full membership.",
              config->pkey_value);
    return UCS_ERR_INVALID_PARAM;
}

static ucs_status_t uct_ib_iface_init_lmc(uct_ib_iface_t *iface,
                                          const uct_ib_iface_config_t *config)
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

    iface->path_bits = ucs_calloc(1, num_path_bits * sizeof(*iface->path_bits),
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

static ucs_status_t uct_ib_iface_set_cqe_size_var(uct_ib_iface_t *iface,
                                                  size_t *inl,
                                                  int *env_var_added)
{
    const char *cqe_size_env_var = "MLX5_CQE_SIZE";
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    const char *cqe_size_env_value;
    size_t cqe_size_min, cqe_size;
    char cqe_size_buf[32];
    int ret;

    cqe_size_min       = (*inl > 32) ? 128 : 64;
    cqe_size_env_value = getenv(cqe_size_env_var);

    if (cqe_size_env_value != NULL) {
        cqe_size = atol(cqe_size_env_value);
        if (cqe_size < cqe_size_min) {
            ucs_error("%s is set to %zu, but at least %zu is required (inl: %zu)",
                      cqe_size_env_var, cqe_size, cqe_size_min, *inl);
            return UCS_ERR_INVALID_PARAM;
        }
        *env_var_added = 0;
    } else {
        /* CQE size is not defined by the environment, set it according to inline
         * size and cache line size.
         */
        cqe_size = ucs_max(cqe_size_min, UCS_SYS_CACHE_LINE_SIZE);
        cqe_size = ucs_max(cqe_size, 64);  /* at least 64 */
        cqe_size = ucs_min(cqe_size, 128); /* at most 128 */
        snprintf(cqe_size_buf, sizeof(cqe_size_buf),"%zu", cqe_size);
        ucs_debug("%s: setting %s=%s", uct_ib_device_name(dev), cqe_size_env_var,
                  cqe_size_buf);
        ret = ibv_exp_setenv(dev->ibv_context, cqe_size_env_var, cqe_size_buf, 1);
        if (ret) {
            ucs_error("ibv_exp_setenv(%s=%s) failed: %m", cqe_size_env_var,
                      cqe_size_buf);
            return UCS_ERR_INVALID_PARAM;
        }
        *env_var_added = 1;
    }
    *inl  = cqe_size / 2;
    return UCS_OK;
}

static void uct_ib_iface_unset_cqe_size_var(uct_ib_device_t *dev,
                                            int env_var_added)
{
    const char *cqe_size_env_var = "MLX5_CQE_SIZE";

    if (env_var_added) {
        /* if we created a new environment variable, remove it */
        if (ibv_exp_unsetenv(dev->ibv_context, cqe_size_env_var)) {
            ucs_warn("unsetenv(%s) failed: %m", cqe_size_env_var);
        }
    }
}

static ucs_status_t uct_ib_iface_create_cq(uct_ib_iface_t *iface, int cq_length,
                                           size_t *inl, int preferred_cpu,
                                           struct ibv_cq **cq_p)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    ucs_status_t status;
    struct ibv_cq *cq;
    int env_var_added;

    status = uct_ib_iface_set_cqe_size_var(iface, inl, &env_var_added);
    if (status != UCS_OK) {
        goto out;
    }

    cq = ibv_create_cq(dev->ibv_context, cq_length, NULL, iface->comp_channel,
                       preferred_cpu);
    if (cq == NULL) {
        ucs_error("ibv_create_cq(cqe=%d) failed: %m", cq_length);
        status = UCS_ERR_IO_ERROR;
        goto out_unsetenv;
    }

    *cq_p = cq;
    status = UCS_OK;

out_unsetenv:
    uct_ib_iface_unset_cqe_size_var(dev, env_var_added);
out:
    return status;
}

#if HAVE_IBV_EX_HW_TM
static ucs_status_t uct_ib_iface_create_cq_ex(uct_ib_iface_t *iface, int cq_length,
                                              size_t *inl, int preferred_cpu,
                                              struct ibv_cq **cq_p)
{
    uct_ib_device_t *dev               = uct_ib_iface_device(iface);
    struct ibv_cq_init_attr_ex cq_attr = {};
    struct ibv_cq_ex *cq;
    ucs_status_t status;
    int env_var_added;

    status = uct_ib_iface_set_cqe_size_var(iface, inl, &env_var_added);
    if (status != UCS_OK) {
        goto out;
    }
    cq_attr.channel     = iface->comp_channel;
    cq_attr.cqe         = cq_length;
    cq_attr.comp_vector = preferred_cpu;
    cq_attr.cq_context  = NULL;
    cq_attr.wc_flags    = IBV_WC_EX_WITH_TM_INFO  |
                          IBV_WC_EX_WITH_BYTE_LEN |
                          IBV_WC_EX_WITH_SRC_QP   |
                          IBV_WC_EX_WITH_QP_NUM   |
                          IBV_WC_EX_WITH_SLID     |
                          IBV_WC_EX_WITH_IMM;

    cq = ibv_create_cq_ex(dev->ibv_context, &cq_attr);
    if (cq == NULL) {
        ucs_error("ibv_create_cq_ex(cqe=%d) failed: %m", cq_length);
        status = UCS_ERR_IO_ERROR;
        goto out_unsetenv;
    }

    *cq_p = ibv_cq_ex_to_cq(cq);
    status = UCS_OK;

out_unsetenv:
    uct_ib_iface_unset_cqe_size_var(dev, env_var_added);
out:
    return status;
}

#endif

static ucs_status_t uct_ib_iface_create_recv_cq(uct_ib_iface_t *iface,
                                                int cq_length,
                                                size_t *inl, int preferred_cpu,
                                                unsigned is_ex,
                                                struct ibv_cq **cq_p)
{
#if HAVE_IBV_EX_HW_TM
    if (is_ex) {
        return uct_ib_iface_create_cq_ex(iface, cq_length, inl,
                                         preferred_cpu, cq_p);
    } else
#endif
    {
        return uct_ib_iface_create_cq(iface, cq_length, inl,
                                      preferred_cpu, cq_p);
    }
}

/**
 * @param rx_headroom   Headroom requested by the user.
 * @param rx_priv_len   Length of transport private data to reserve (0 if unused)
 * @param rx_hdr_len    Length of transport network header.
 * @param mss           Maximal segment size (transport limit).
 */
UCS_CLASS_INIT_FUNC(uct_ib_iface_t, uct_ib_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    unsigned rx_priv_len, unsigned rx_hdr_len,
                    unsigned tx_cq_len, unsigned rx_cq_len, size_t mss,
                    unsigned is_ex_recv_cq, const uct_ib_iface_config_t *config)
{
    uct_ib_device_t *dev = &ucs_derived_of(md, uct_ib_md_t)->dev;
    int preferred_cpu = ucs_cpu_set_find_lcs(&params->cpu_mask);
    ucs_status_t status;
    uint8_t port_num;
    size_t inl;

    if (params->stats_root == NULL) {
        UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &ops->super, md, worker,
                                  params, &config->super
                                  UCS_STATS_ARG(dev->stats)
                                  UCS_STATS_ARG(params->dev_name));
    } else {
        UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &ops->super, md, worker,
                                  params, &config->super
                                  UCS_STATS_ARG(params->stats_root)
                                  UCS_STATS_ARG(params->dev_name));
    }

    status = uct_ib_device_find_port(dev, params->dev_name, &port_num);
    if (status != UCS_OK) {
        goto err;
    }

    self->ops                      = ops;

    self->config.rx_payload_offset = sizeof(uct_ib_iface_recv_desc_t) +
                                     ucs_max(sizeof(uct_recv_desc_t) +
                                             params->rx_headroom,
                                             rx_priv_len + rx_hdr_len);
    self->config.rx_hdr_offset     = self->config.rx_payload_offset - rx_hdr_len;
    self->config.rx_headroom_offset= self->config.rx_payload_offset -
                                     params->rx_headroom;
    self->config.seg_size          = ucs_min(mss, config->super.max_bcopy);
    self->config.tx_max_poll       = config->tx.max_poll;
    self->config.rx_max_poll       = config->rx.max_poll;
    self->config.rx_max_batch      = ucs_min(config->rx.max_batch,
                                             config->rx.queue_len / 4);
    self->config.port_num          = port_num;
    self->config.sl                = config->sl;
    self->config.gid_index         = config->gid_index;
    self->release_desc.cb          = uct_ib_iface_release_desc;

    status = uct_ib_iface_init_pkey(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_device_query_gid(dev, self->config.port_num,
                                     self->config.gid_index, &self->gid);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_iface_init_lmc(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    self->comp_channel = ibv_create_comp_channel(dev->ibv_context);
    if (self->comp_channel == NULL) {
        ucs_error("ibv_create_comp_channel() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_free_path_bits;
    }

    status = ucs_sys_fcntl_modfl(self->comp_channel->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_destroy_comp_channel;
    }

    inl = config->rx.inl;
    status = uct_ib_iface_create_cq(self, tx_cq_len, &inl, preferred_cpu,
                                    &self->send_cq);
    if (status != UCS_OK) {
        goto err_destroy_comp_channel;
    }
    ucs_assert_always(inl <= UINT8_MAX);
    self->config.max_inl_resp = inl;

    inl = config->rx.inl;
    status = uct_ib_iface_create_recv_cq(self, rx_cq_len, &inl,
                                         preferred_cpu, is_ex_recv_cq,
                                         &self->recv_cq);
    if (status != UCS_OK) {
        goto err_destroy_send_cq;
    }

    /* Address scope and size */
    if (config->addr_type == UCT_IB_IFACE_ADDRESS_TYPE_AUTO) {
        if (IBV_PORT_IS_LINK_LAYER_ETHERNET(uct_ib_iface_port_attr(self))) {
            self->addr_type = UCT_IB_ADDRESS_TYPE_ETH;
        } else {
            self->addr_type = uct_ib_address_scope(self->gid.global.subnet_prefix);
        }
    } else {
        ucs_assert(config->addr_type < UCT_IB_ADDRESS_TYPE_LAST);
        self->addr_type = config->addr_type;
    }

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
    cpu_set_t temp_cpu_mask, process_affinity;
    int ret;
    uct_ib_md_t *md = uct_ib_iface_md(iface);
    
    active_width = uct_ib_iface_port_attr(iface)->active_width;
    active_speed = uct_ib_iface_port_attr(iface)->active_speed;
    active_mtu   = uct_ib_iface_port_attr(iface)->active_mtu;

    /* Get active width */
    if (!ucs_is_pow2(active_width) ||
        (active_width < 1) || (ucs_ilog2(active_width) > 3))
    {
        ucs_error("Invalid active_width on %s:%d: %d",
                  UCT_IB_IFACE_ARG(iface), active_width);
        return UCS_ERR_IO_ERROR;
    }

    memset(iface_attr, 0, sizeof(*iface_attr));

    iface_attr->device_addr_len = iface->addr_size;

    switch (active_speed) {
    case 1: /* SDR */
        iface_attr->latency.overhead = 5000e-9;
        signal_rate                  = 2.5e9;
        encoding                     = 8.0/10.0;
        break;
    case 2: /* DDR */
        iface_attr->latency.overhead = 2500e-9;
        signal_rate                  = 5.0e9;
        encoding                     = 8.0/10.0;
        break;
    case 4:
        iface_attr->latency.overhead = 1300e-9;
        if (IBV_PORT_IS_LINK_LAYER_ETHERNET(uct_ib_iface_port_attr(iface))) {
            /* 10/40g Eth  */
            signal_rate              = 10.3125e9;
            encoding                 = 64.0/66.0;
        } else {
            /* QDR */
            signal_rate              = 10.0e9;
            encoding                 = 8.0/10.0;
        }
        break;
    case 8: /* FDR10 */
        iface_attr->latency.overhead = 700e-9;
        signal_rate                  = 10.3125e9;
        encoding                     = 64.0/66.0;
        break;
    case 16: /* FDR */
        iface_attr->latency.overhead = 700e-9;
        signal_rate                  = 14.0625e9;
        encoding                     = 64.0/66.0;
        break;
    case 32: /* EDR */
        iface_attr->latency.overhead = 600e-9;
        signal_rate                  = 25.78125e9;
        encoding                     = 64.0/66.0;
        break;
    default:
        ucs_error("Invalid active_speed on %s:%d: %d",
                  UCT_IB_IFACE_ARG(iface), active_speed);
        return UCS_ERR_IO_ERROR;
    }

    if (md->config.prefer_nearest_device) {
        ret = sched_getaffinity(0, sizeof(process_affinity), &process_affinity);
        if (ret) {
            ucs_error("sched_getaffinity() failed: %m");
            return UCS_ERR_INVALID_PARAM;
        }

        /* update latency for remote device */
        CPU_AND(&temp_cpu_mask, &dev->local_cpus, &process_affinity);
        if (!CPU_EQUAL(&process_affinity, &temp_cpu_mask)) {
            iface_attr->latency.overhead += 200e-9;
        }
    }
    
    iface_attr->latency.growth = 0;

    /* Wire speed calculation: Width * SignalRate * Encoding */
    width                 = ib_port_widths[ucs_ilog2(active_width)];
    wire_speed            = (width * signal_rate * encoding) / 8.0;

    /* Calculate packet overhead  */
    mtu                   = ucs_min(uct_ib_mtu_value(active_mtu),
                                    iface->config.seg_size);

    extra_pkt_len = UCT_IB_BTH_LEN + xport_hdr_len +  UCT_IB_ICRC_LEN + UCT_IB_VCRC_LEN + UCT_IB_DELIM_LEN;

    if (IBV_PORT_IS_LINK_LAYER_ETHERNET(uct_ib_iface_port_attr(iface))) {
        extra_pkt_len += UCT_IB_GRH_LEN + UCT_IB_ROCE_LEN;
    } else {
        /* TODO check if UCT_IB_DELIM_LEN is present in RoCE as well */
        extra_pkt_len += UCT_IB_LRH_LEN;
    }

    iface_attr->bandwidth = (wire_speed * mtu) / (mtu + extra_pkt_len);
    iface_attr->priority  = uct_ib_device_spec(dev)->priority;

    return UCS_OK;
}

ucs_status_t uct_ib_iface_wakeup_arm(uct_wakeup_h wakeup)
{
    int res, send_cq_count = 0, recv_cq_count = 0;
    ucs_status_t status;
    struct ibv_cq *cq;
    void *cq_context;
    uct_ib_iface_t *iface = ucs_derived_of(wakeup->iface, uct_ib_iface_t);

    do {
        res = ibv_get_cq_event(iface->comp_channel, &cq, &cq_context);
        if (0 == res) {
            if (iface->send_cq == cq) {
                ++send_cq_count;
            }
            if (iface->recv_cq == cq) {
                ++recv_cq_count;
            }
        }
    } while (res == 0);

    if (errno != EAGAIN) {
        return UCS_ERR_IO_ERROR;
    }

    if (send_cq_count > 0) {
        ibv_ack_cq_events(iface->send_cq, send_cq_count);
    }

    if (recv_cq_count > 0) {
        ibv_ack_cq_events(iface->recv_cq, recv_cq_count);
    }

    /* avoid re-arming the interface if any events exists */
    if ((send_cq_count > 0) || (recv_cq_count > 0)) {
        return UCS_ERR_BUSY;
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
    ucs_status_t status;
    int res;
    struct pollfd polled = { .fd = wakeup->fd, .events = POLLIN };

    status = wakeup->iface->ops.iface_wakeup_arm(wakeup);
    if (UCS_ERR_BUSY == status) { /* if UCS_ERR_BUSY returned - no poll() must called */
        return UCS_OK;
    } else if (status != UCS_OK) {
        return status;
    }

    do {
        res = poll(&polled, 1, -1);
    } while ((res == -1) && (errno == EINTR));

    if ((res != 1) || (polled.revents != POLLIN)) {
        return UCS_ERR_IO_ERROR;
    }

    return status;
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
        ucs_error("ibv_req_notify_cq("UCT_IB_IFACE_FMT", cq) failed: %m",
                  UCT_IB_IFACE_ARG(iface));
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
