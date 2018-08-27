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
#include <ucs/time/time.h>
#include <ucs/sys/numa.h>
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
   "Maximum number of send WQEs which can be posted without requesting a completion.",
   ucs_offsetof(uct_ib_iface_config_t, tx.cq_moderation), UCS_CONFIG_TYPE_UINT},

#if HAVE_DECL_IBV_EXP_CQ_MODERATION
  {"TX_EVENT_MOD_COUNT", "0",
   "Number of send completions for which an event would be generated (0 - disabled).",
   ucs_offsetof(uct_ib_iface_config_t, tx.cq_moderation_count), UCS_CONFIG_TYPE_UINT},

  {"TX_EVENT_MOD_PERIOD", "0us",
   "Time period to generate send event (0 - disabled).",
   ucs_offsetof(uct_ib_iface_config_t, tx.cq_moderation_period), UCS_CONFIG_TYPE_TIME},

  {"RX_EVENT_MOD_COUNT", "0",
   "Number of received messages for which an event would be generated (0 - disabled).",
   ucs_offsetof(uct_ib_iface_config_t, rx.cq_moderation_count), UCS_CONFIG_TYPE_UINT},

  {"RX_EVENT_MOD_PERIOD", "0us",
   "Time period to generate receive event (0 - disabled).",
   ucs_offsetof(uct_ib_iface_config_t, rx.cq_moderation_period), UCS_CONFIG_TYPE_TIME},
#endif /* HAVE_DECL_IBV_EXP_CQ_MODERATION */

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

  {"RX_MAX_POLL", "16",
   "Max number of receive completions to pick during RX poll",
   ucs_offsetof(uct_ib_iface_config_t, rx.max_poll), UCS_CONFIG_TYPE_UINT},

  {"RX_INLINE", "0",
   "Number of bytes to request for inline receive. If the maximal supported size\n"
   "is smaller, it will be used instead. If it is possible to support a larger\n"
   "size than requested with the same hardware resources, it will be used instead.",
   ucs_offsetof(uct_ib_iface_config_t, rx.inl), UCS_CONFIG_TYPE_MEMUNITS},

  UCT_IFACE_MPOOL_CONFIG_FIELDS("RX_", -1, 0, "receive",
                                ucs_offsetof(uct_ib_iface_config_t, rx.mp), ""),

  {"ADDR_TYPE", "auto",
   "Set the interface address type. \"auto\" mode detects the type according to\n"
   "link layer type and IB subnet prefix.",
   ucs_offsetof(uct_ib_iface_config_t, addr_type),
   UCS_CONFIG_TYPE_ENUM(uct_ib_iface_addr_types)},

  {"SL", "0",
   "IB Service Level / RoCEv2 Ethernet Priority.\n",
   ucs_offsetof(uct_ib_iface_config_t, sl), UCS_CONFIG_TYPE_UINT},

  {"TRAFFIC_CLASS", "0",
   "IB Traffic Class / RoCEv2 Differentiated Services Code Point (DSCP)\n",
   ucs_offsetof(uct_ib_iface_config_t, traffic_class), UCS_CONFIG_TYPE_UINT},

  {"HOP_LIMIT", "255",
   "IB Hop limit / RoCEv2 Time to Live. Should be between 0 and 255.\n",
   ucs_offsetof(uct_ib_iface_config_t, hop_limit), UCS_CONFIG_TYPE_UINT},

  {"LID_PATH_BITS", "0-17",
   "List of IB Path bits separated by comma (a,b,c) "
   "which will be the low portion of the LID, according to the LMC in the fabric.",
   ucs_offsetof(uct_ib_iface_config_t, lid_path_bits), UCS_CONFIG_TYPE_ARRAY(path_bits_spec)},

  {"PKEY", "0x7fff",
   "Which pkey value to use. Should be between 0 and 0x7fff.",
   ucs_offsetof(uct_ib_iface_config_t, pkey_value), UCS_CONFIG_TYPE_HEX},

#if HAVE_IBV_EXP_RES_DOMAIN
  {"RESOURCE_DOMAIN", "y",
   "Enable multiple resource domains (experimental).",
   ucs_offsetof(uct_ib_iface_config_t, enable_res_domain), UCS_CONFIG_TYPE_BOOL},
#endif


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
                                &config->rx.mp, grow,
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
    int is_local_ib;

    uct_ib_address_unpack(ib_addr, &lid, &is_global, &gid);

    ucs_assert(iface->addr_type < UCT_IB_ADDRESS_TYPE_LAST);
    switch (iface->addr_type) {
    case UCT_IB_ADDRESS_TYPE_LINK_LOCAL:
    case UCT_IB_ADDRESS_TYPE_SITE_LOCAL:
    case UCT_IB_ADDRESS_TYPE_GLOBAL:
         is_local_ib = 1;
         break;
    case UCT_IB_ADDRESS_TYPE_ETH:
         is_local_ib = 0;
         break;
    default:
         ucs_fatal("Unknown address type %d", iface->addr_type);
         break;
    }

    if (is_local_ib && (ib_addr->flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_IB)) {
        /* same subnet prefix */
        return gid.global.subnet_prefix == iface->gid.global.subnet_prefix;
    } else if (!is_local_ib && (ib_addr->flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH)) {
        /* there shouldn't be a lid and the gid flag should be on */
        ucs_assert(ib_addr->flags & UCT_IB_ADDRESS_FLAG_GID);
        ucs_assert(!(ib_addr->flags & UCT_IB_ADDRESS_FLAG_LID));
        return 1;
    } else {
        /* local and remote have different link layers and therefore are unreachable */
        return 0;
    }
}

ucs_status_t uct_ib_iface_create_ah(uct_ib_iface_t *iface,
                                    struct ibv_ah_attr *ah_attr,
                                    struct ibv_ah **ah_p)
{
    struct ibv_ah *ah;
    char buf[128];
    char *p, *endp;

    ah = ibv_create_ah(uct_ib_iface_md(iface)->pd, ah_attr);

    if (ah == NULL) {
        p    = buf;
        endp = buf + sizeof(buf);
        snprintf(p, endp - p, "dlid=%d sl=%d port=%d src_path_bits=%d",
                 ah_attr->dlid, ah_attr->sl,
                 ah_attr->port_num, ah_attr->src_path_bits);
        p += strlen(p);

        if (ah_attr->is_global) {
            snprintf(p, endp - p, " dgid=");
            p += strlen(p);
            inet_ntop(AF_INET6, &ah_attr->grh.dgid, p, endp - p);
            p += strlen(p);
            snprintf(p, endp - p, " sgid_index=%d traffic_class=%d",
                     ah_attr->grh.sgid_index, ah_attr->grh.traffic_class);
        }

        ucs_error("ibv_create_ah(%s) on "UCT_IB_IFACE_FMT" failed: %m", buf,
                  UCT_IB_IFACE_ARG(iface));
        return UCS_ERR_INVALID_ADDR;
    }

    *ah_p        = ah;
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
        num_path_bits += 1 + abs((int)(config->lid_path_bits.ranges[i].first -
                                       config->lid_path_bits.ranges[i].last));
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

#if HAVE_DECL_IBV_EXP_SETENV
static int uct_ib_max_cqe_size()
{
    static int max_cqe_size = -1;

    if (max_cqe_size == -1) {
#ifdef __aarch64__
        char arm_board_vendor[128];
        ucs_aarch64_cpuid_t cpuid;
        ucs_aarch64_cpuid(&cpuid);

        arm_board_vendor[0] = '\0';
        ucs_read_file(arm_board_vendor, sizeof(arm_board_vendor), 1,
                      "/sys/devices/virtual/dmi/id/board_vendor");
        ucs_debug("arm_board_vendor is '%s'", arm_board_vendor);

        max_cqe_size = ((strcasestr(arm_board_vendor, "Huawei")) &&
                        (cpuid.implementer == 0x41) && (cpuid.architecture == 8) &&
                        (cpuid.variant == 0)        && (cpuid.part == 0xd08)     &&
                        (cpuid.revision == 2))
                       ? 64 : 128;
#else
        max_cqe_size = 128;
#endif
        ucs_debug("max IB CQE size is %d", max_cqe_size);
    }

    return max_cqe_size;
}
#endif

struct ibv_cq *uct_ib_create_cq(struct ibv_context *context, int cqe,
                                struct ibv_comp_channel *channel,
                                int comp_vector, int ignore_overrun)
{
    struct ibv_cq *cq;
#if HAVE_DECL_IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN
    struct ibv_cq_init_attr_ex cq_attr = {};

    cq_attr.cqe = cqe;
    cq_attr.channel = channel;
    cq_attr.comp_vector = comp_vector;
    if (ignore_overrun) {
        cq_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
        cq_attr.flags = IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN;
    }

    cq = ibv_cq_ex_to_cq(ibv_create_cq_ex(context, &cq_attr));
#else
    cq = ibv_create_cq(context, cqe, NULL, channel, comp_vector);
#endif
    return cq;
}

static ucs_status_t uct_ib_iface_create_cq(uct_ib_iface_t *iface, int cq_length,
                                           size_t *inl, int preferred_cpu,
                                           int flags, struct ibv_cq **cq_p)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    struct ibv_cq *cq;
    size_t cqe_size = 64;
    ucs_status_t status;
#if HAVE_DECL_IBV_EXP_SETENV
    static const char *cqe_size_env_var = "MLX5_CQE_SIZE";
    const char *cqe_size_env_value;
    size_t cqe_size_min;
    char cqe_size_buf[32];
    int env_var_added = 0;
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
    } else {
        /* CQE size is not defined by the environment, set it according to inline
         * size and cache line size.
         */
        cqe_size = ucs_max(cqe_size_min, UCS_SYS_CACHE_LINE_SIZE);
        cqe_size = ucs_max(cqe_size, 64);  /* at least 64 */
        cqe_size = ucs_min(cqe_size, uct_ib_max_cqe_size());
        snprintf(cqe_size_buf, sizeof(cqe_size_buf),"%zu", cqe_size);
        ucs_debug("%s: setting %s=%s", uct_ib_device_name(dev), cqe_size_env_var,
                  cqe_size_buf);
        ret = ibv_exp_setenv(dev->ibv_context, cqe_size_env_var, cqe_size_buf, 1);
        if (ret) {
            ucs_error("ibv_exp_setenv(%s=%s) failed: %m", cqe_size_env_var,
                      cqe_size_buf);
            return UCS_ERR_INVALID_PARAM;
        }

        env_var_added = 1;
    }
#endif
    cq = uct_ib_create_cq(dev->ibv_context, cq_length, iface->comp_channel,
                          preferred_cpu, flags & UCT_IB_CQ_IGNORE_OVERRUN);
    if (cq == NULL) {
        ucs_error("ibv_create_cq(cqe=%d) failed: %m", cq_length);
        status = UCS_ERR_IO_ERROR;
        goto out_unsetenv;
    }

    *cq_p = cq;
    *inl  = cqe_size / 2;
    status = UCS_OK;

out_unsetenv:
#if HAVE_DECL_IBV_EXP_SETENV
    if (env_var_added) {
        /* if we created a new environment variable, remove it */
        ret = ibv_exp_unsetenv(dev->ibv_context, cqe_size_env_var);
        if (ret) {
            ucs_warn("unsetenv(%s) failed: %m", cqe_size_env_var);
        }
    }
#endif
    return status;
}


static ucs_status_t uct_ib_iface_set_moderation(struct ibv_cq *cq,
                                                unsigned count, double period_usec)
{
#if HAVE_DECL_IBV_EXP_CQ_MODERATION
    unsigned period = (unsigned)(period_usec * UCS_USEC_PER_SEC);

    if (count > UINT16_MAX) {
        ucs_error("CQ moderation count is too high: %u, max value: %u", count, UINT16_MAX);
        return UCS_ERR_INVALID_PARAM;
    } else if (count == 0) {
        /* in case if count value is 0 (unchanged default value) - set it to maximum
         * possible value */
        count = UINT16_MAX;
    }

    if (period > UINT16_MAX) {
        ucs_error("CQ moderation period is too high: %u, max value: %uus", period, UINT16_MAX);
        return UCS_ERR_INVALID_PARAM;
    } else if (period == 0) {
        /* in case if count value is 0 (unchanged default value) - set it to maximum
         * possible value, the same behavior as counter */
        period = UINT16_MAX;
    }

    if ((count < UINT16_MAX) || (period < UINT16_MAX)) {
        struct ibv_exp_cq_attr cq_attr = {
            .comp_mask            = IBV_EXP_CQ_ATTR_MODERATION,
            .moderation.cq_count  = (uint16_t)(count),
            .moderation.cq_period = (uint16_t)(period),
            .cq_cap_flags         = 0
        };
        if (ibv_exp_modify_cq(cq, &cq_attr, IBV_EXP_CQ_MODERATION)) {
            ucs_error("ibv_exp_modify_cq(count=%d, period=%d) failed: %m", count, period);
            return UCS_ERR_IO_ERROR;
        }
    }
#endif /* HAVE_DECL_IBV_EXP_CQ_MODERATION */

    return UCS_OK;
}

static int uct_ib_iface_res_domain_cmp(uct_ib_iface_res_domain_t *res_domain,
                                       uct_ib_iface_t *iface)
{
#if HAVE_IBV_EXP_RES_DOMAIN
    uct_ib_device_t *dev = uct_ib_iface_device(iface);

    return res_domain->ibv_domain->context == dev->ibv_context;
#elif HAVE_DECL_IBV_ALLOC_TD
    uct_ib_md_t     *md  = uct_ib_iface_md(iface);

    return res_domain->pd == md->pd;
#else
    return 1;
#endif
}

static ucs_status_t
uct_ib_iface_res_domain_init(uct_ib_iface_res_domain_t *res_domain,
                             uct_ib_iface_t *iface)
{
#if HAVE_IBV_EXP_RES_DOMAIN
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    struct ibv_exp_res_domain_init_attr attr;

    attr.comp_mask    = IBV_EXP_RES_DOMAIN_THREAD_MODEL |
                        IBV_EXP_RES_DOMAIN_MSG_MODEL;
    attr.msg_model    = IBV_EXP_MSG_LOW_LATENCY;

    switch (iface->super.worker->thread_mode) {
    case UCS_THREAD_MODE_SINGLE:
        attr.thread_model = IBV_EXP_THREAD_SINGLE;
        break;
    case UCS_THREAD_MODE_SERIALIZED:
        attr.thread_model = IBV_EXP_THREAD_UNSAFE;
        break;
    default:
        attr.thread_model = IBV_EXP_THREAD_SAFE;
        break;
    }

    res_domain->ibv_domain = ibv_exp_create_res_domain(dev->ibv_context, &attr);
    if (res_domain->ibv_domain == NULL) {
        ucs_error("ibv_exp_create_res_domain() on %s failed: %m",
                  uct_ib_device_name(dev));
        return UCS_ERR_IO_ERROR;
    }
#elif HAVE_DECL_IBV_ALLOC_TD
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    uct_ib_md_t     *md  = uct_ib_iface_md(iface);
    struct ibv_parent_domain_init_attr attr;
    struct ibv_td_init_attr td_attr;

    if (iface->super.worker->thread_mode == UCS_THREAD_MODE_MULTI) {
        td_attr.comp_mask = 0;
        res_domain->td = ibv_alloc_td(dev->ibv_context, &td_attr);
        if (res_domain->td == NULL) {
            ucs_error("ibv_alloc_td() on %s failed: %m",
                      uct_ib_device_name(dev));
            return UCS_ERR_IO_ERROR;
        }
    } else {
        res_domain->td = NULL;
        res_domain->ibv_domain = NULL;
        res_domain->pd = md->pd;
        return UCS_OK;
    }

    attr.td = res_domain->td;
    attr.pd = md->pd;
    attr.comp_mask = 0;
    res_domain->ibv_domain = ibv_alloc_parent_domain(dev->ibv_context, &attr);
    if (res_domain->ibv_domain == NULL) {
        ucs_error("ibv_alloc_parent_domain() on %s failed: %m",
                  uct_ib_device_name(dev));
        ibv_dealloc_td(res_domain->td);
        return UCS_ERR_IO_ERROR;
    }
    res_domain->pd = md->pd;
#endif
    return UCS_OK;
}

static void uct_ib_iface_res_domain_cleanup(uct_ib_iface_res_domain_t *res_domain)
{
#if HAVE_IBV_EXP_RES_DOMAIN
    struct ibv_exp_destroy_res_domain_attr attr;
    int ret;

    attr.comp_mask = 0;
    ret = ibv_exp_destroy_res_domain(res_domain->ibv_domain->context,
                                     res_domain->ibv_domain, &attr);
    if (ret != 0) {
        ucs_warn("ibv_exp_destroy_res_domain() failed: %m");
    }
#elif HAVE_DECL_IBV_ALLOC_TD
    int ret;

    if (res_domain->ibv_domain != NULL) {
        ret = ibv_dealloc_pd(res_domain->ibv_domain);
        if (ret != 0) {
            ucs_warn("ibv_dealloc_pd() failed: %m");
            return;
        }

        ret = ibv_dealloc_td(res_domain->td);
        if (ret != 0) {
            ucs_warn("ibv_dealloc_td() failed: %m");
        }
    }
#endif
}

UCS_CLASS_INIT_FUNC(uct_ib_iface_t, uct_ib_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_ib_iface_config_t *config,
                    const uct_ib_iface_init_attr_t *init_attr)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);
    uct_ib_device_t *dev = &ib_md->dev;
    int preferred_cpu = ucs_cpu_set_find_lcs(&params->cpu_mask);
    ucs_status_t status;
    uint8_t port_num;
    size_t inl;

    ucs_assert(params->open_mode & UCT_IFACE_OPEN_MODE_DEVICE);

    if (params->stats_root == NULL) {
        UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &ops->super, md, worker,
                                  params, &config->super
                                  UCS_STATS_ARG(dev->stats)
                                  UCS_STATS_ARG(params->mode.device.dev_name));
    } else {
        UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &ops->super, md, worker,
                                  params, &config->super
                                  UCS_STATS_ARG(params->stats_root)
                                  UCS_STATS_ARG(params->mode.device.dev_name));
    }

    status = uct_ib_device_find_port(dev, params->mode.device.dev_name, &port_num);
    if (status != UCS_OK) {
        goto err;
    }

    self->ops                       = ops;

    self->config.rx_payload_offset  = sizeof(uct_ib_iface_recv_desc_t) +
                                      ucs_max(sizeof(uct_recv_desc_t) +
                                              params->rx_headroom,
                                              init_attr->rx_priv_len +
                                              init_attr->rx_hdr_len);
    self->config.rx_hdr_offset      = self->config.rx_payload_offset -
                                      init_attr->rx_hdr_len;
    self->config.rx_headroom_offset = self->config.rx_payload_offset -
                                      params->rx_headroom;
    self->config.seg_size           = init_attr->seg_size;
    self->config.tx_max_poll        = config->tx.max_poll;
    self->config.rx_max_poll        = config->rx.max_poll;
    self->config.rx_max_batch       = ucs_min(config->rx.max_batch,
                                              config->rx.queue_len / 4);
    self->config.port_num           = port_num;
    self->config.sl                 = config->sl;
    self->config.traffic_class      = config->traffic_class;
    self->config.hop_limit          = config->hop_limit;
    self->release_desc.cb           = uct_ib_iface_release_desc;

    self->config.enable_res_domain  = config->enable_res_domain;

    status = uct_ib_iface_init_pkey(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_device_select_gid_index(dev, self->config.port_num,
                                            ib_md->config.gid_index,
                                            &self->config.gid_index);
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

    if ((init_attr->res_domain_key == UCT_IB_IFACE_NULL_RES_DOMAIN_KEY) ||
        !self->config.enable_res_domain) {
        self->res_domain = NULL;
    } else {
        self->res_domain = uct_worker_tl_data_get(self->super.worker,
                                                  init_attr->res_domain_key,
                                                  uct_ib_iface_res_domain_t,
                                                  uct_ib_iface_res_domain_cmp,
                                                  uct_ib_iface_res_domain_init,
                                                  self);
        if (UCS_PTR_IS_ERR(self->res_domain)) {
            status = UCS_PTR_STATUS(self->res_domain);
            goto err_free_path_bits;
        }
    }

    self->comp_channel = ibv_create_comp_channel(dev->ibv_context);
    if (self->comp_channel == NULL) {
        ucs_error("ibv_create_comp_channel() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_put_res_domain;
    }

    status = ucs_sys_fcntl_modfl(self->comp_channel->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_destroy_comp_channel;
    }

    inl = config->rx.inl;
    status = uct_ib_iface_create_cq(self, init_attr->tx_cq_len, &inl,
                                    preferred_cpu, init_attr->flags,
                                    &self->cq[UCT_IB_DIR_TX]);
    if (status != UCS_OK) {
        goto err_destroy_comp_channel;
    }
    ucs_assert_always(inl <= UINT8_MAX);
    self->config.max_inl_resp = inl;

    status = uct_ib_iface_set_moderation(self->cq[UCT_IB_DIR_TX],
                                         config->tx.cq_moderation_count,
                                         config->tx.cq_moderation_period);
    if (status != UCS_OK) {
        goto err_destroy_send_cq;
    }

    inl = config->rx.inl;
    status = uct_ib_iface_create_cq(self, init_attr->rx_cq_len, &inl,
                                    preferred_cpu, init_attr->flags,
                                    &self->cq[UCT_IB_DIR_RX]);
    if (status != UCS_OK) {
        goto err_destroy_send_cq;
    }

    status = uct_ib_iface_set_moderation(self->cq[UCT_IB_DIR_RX],
                                         config->rx.cq_moderation_count,
                                         config->rx.cq_moderation_period);
    if (status != UCS_OK) {
        goto err_destroy_recv_cq;
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

err_destroy_recv_cq:
    ibv_destroy_cq(self->cq[UCT_IB_DIR_RX]);
err_destroy_send_cq:
    ibv_destroy_cq(self->cq[UCT_IB_DIR_TX]);
err_destroy_comp_channel:
    ibv_destroy_comp_channel(self->comp_channel);
err_put_res_domain:
    if (self->res_domain != NULL) {
        uct_worker_tl_data_put(self->res_domain, uct_ib_iface_res_domain_cleanup);
    }
err_free_path_bits:
    ucs_free(self->path_bits);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ib_iface_t)
{
    int ret;

    ret = ibv_destroy_cq(self->cq[UCT_IB_DIR_RX]);
    if (ret != 0) {
        ucs_warn("ibv_destroy_cq(recv_cq) returned %d: %m", ret);
    }

    ret = ibv_destroy_cq(self->cq[UCT_IB_DIR_TX]);
    if (ret != 0) {
        ucs_warn("ibv_destroy_cq(send_cq) returned %d: %m", ret);
    }

    ret = ibv_destroy_comp_channel(self->comp_channel);
    if (ret != 0) {
        ucs_warn("ibv_destroy_comp_channel(comp_channel) returned %d: %m", ret);
    }

    if (self->res_domain != NULL) {
        uct_worker_tl_data_put(self->res_domain, uct_ib_iface_res_domain_cleanup);
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

static ucs_status_t uct_ib_iface_get_numa_latency(uct_ib_iface_t *iface,
                                                  double *latency)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    uct_ib_md_t *md      = uct_ib_iface_md(iface);
    cpu_set_t temp_cpu_mask, process_affinity;
#if HAVE_NUMA
    int distance, min_cpu_distance;
    int cpu, num_cpus;
#endif
    int ret;

    if (!md->config.prefer_nearest_device) {
        *latency = 0;
        return UCS_OK;
    }

    ret = sched_getaffinity(0, sizeof(process_affinity), &process_affinity);
    if (ret) {
        ucs_error("sched_getaffinity() failed: %m");
        return UCS_ERR_INVALID_PARAM;
    }

#if HAVE_NUMA
    /* Try to estimate the extra device latency according to NUMA distance */
    if (dev->numa_node != -1) {
        min_cpu_distance = INT_MAX;
        num_cpus         = ucs_min(CPU_SETSIZE, numa_num_configured_cpus());
        for (cpu = 0; cpu < num_cpus; ++cpu) {
            if (!CPU_ISSET(cpu, &process_affinity)) {
                continue;
            }
            distance = numa_distance(ucs_numa_node_of_cpu(cpu), dev->numa_node);
            if (distance >= UCS_NUMA_MIN_DISTANCE) {
                min_cpu_distance = ucs_min(min_cpu_distance, distance);
            }
        }

        if (min_cpu_distance != INT_MAX) {
            /* set the extra latency to (numa_distance - 10) * 20nsec */
            *latency = (min_cpu_distance - UCS_NUMA_MIN_DISTANCE) * 20e-9;
            return UCS_OK;
        }
    }
#endif

    /* Estimate the extra device latency according to its local CPUs mask */
    CPU_AND(&temp_cpu_mask, &dev->local_cpus, &process_affinity);
    if (CPU_EQUAL(&process_affinity, &temp_cpu_mask)) {
        *latency = 0;
    } else {
        *latency = 200e-9;
    }
    return UCS_OK;
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
    ucs_status_t status;
    double numa_latency;
    
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
    case 32: /* EDR / 100g Eth */
        iface_attr->latency.overhead = 600e-9;
        signal_rate                  = 25.78125e9;
        encoding                     = 64.0/66.0;
        break;
    case 64: /* 50g Eth */
        iface_attr->latency.overhead = 600e-9;
        signal_rate                  = 25.78125e9 * 2;
        encoding                     = 64.0/66.0;
        break;
    default:
        ucs_error("Invalid active_speed on %s:%d: %d",
                  UCT_IB_IFACE_ARG(iface), active_speed);
        return UCS_ERR_IO_ERROR;
    }

    status = uct_ib_iface_get_numa_latency(iface, &numa_latency);
    if (status != UCS_OK) {
        return status;
    }

    iface_attr->latency.overhead += numa_latency;
    iface_attr->latency.growth    = 0;

    /* Wire speed calculation: Width * SignalRate * Encoding */
    width                 = ib_port_widths[ucs_ilog2(active_width)];
    wire_speed            = (width * signal_rate * encoding) / 8.0;

    /* Calculate packet overhead  */
    mtu                   = ucs_min(uct_ib_mtu_value(active_mtu),
                                    iface->config.seg_size);

    extra_pkt_len = UCT_IB_BTH_LEN + xport_hdr_len +  UCT_IB_ICRC_LEN + UCT_IB_VCRC_LEN + UCT_IB_DELIM_LEN;

    if (IBV_PORT_IS_LINK_LAYER_ETHERNET(uct_ib_iface_port_attr(iface))) {
        extra_pkt_len += UCT_IB_GRH_LEN + UCT_IB_ROCE_LEN;
        iface_attr->latency.overhead += 200e-9;
    } else {
        /* TODO check if UCT_IB_DELIM_LEN is present in RoCE as well */
        extra_pkt_len += UCT_IB_LRH_LEN;
    }

    iface_attr->bandwidth = (wire_speed * mtu) / (mtu + extra_pkt_len);
    iface_attr->priority  = uct_ib_device_spec(dev)->priority;

    return UCS_OK;
}

ucs_status_t uct_ib_iface_event_fd_get(uct_iface_h tl_iface, int *fd_p)
{
    uct_ib_iface_t *iface = ucs_derived_of(tl_iface, uct_ib_iface_t);
    *fd_p  = iface->comp_channel->fd;
    return UCS_OK;
}

ucs_status_t uct_ib_iface_pre_arm(uct_ib_iface_t *iface)
{
    int res, send_cq_count, recv_cq_count;
    struct ibv_cq *cq;
    void *cq_context;

    send_cq_count = 0;
    recv_cq_count = 0;
    do {
        res = ibv_get_cq_event(iface->comp_channel, &cq, &cq_context);
        if (0 == res) {
            if (iface->cq[UCT_IB_DIR_TX] == cq) {
                iface->ops->event_cq(iface, UCT_IB_DIR_TX);
                ++send_cq_count;
            }
            if (iface->cq[UCT_IB_DIR_RX] == cq) {
                iface->ops->event_cq(iface, UCT_IB_DIR_RX);
                ++recv_cq_count;
            }
        }
    } while (res == 0);

    if (errno != EAGAIN) {
        return UCS_ERR_IO_ERROR;
    }

    if (send_cq_count > 0) {
        ibv_ack_cq_events(iface->cq[UCT_IB_DIR_TX], send_cq_count);
    }

    if (recv_cq_count > 0) {
        ibv_ack_cq_events(iface->cq[UCT_IB_DIR_RX], recv_cq_count);
    }

    /* avoid re-arming the interface if any events exists */
    if ((send_cq_count > 0) || (recv_cq_count > 0)) {
        ucs_trace("arm_cq: got %d send and %d recv events, returning BUSY",
                  send_cq_count, recv_cq_count);
        return UCS_ERR_BUSY;
    }

    return UCS_OK;
}

ucs_status_t uct_ib_iface_arm_cq(uct_ib_iface_t *iface,
                                 uct_ib_dir_t dir,
                                 int solicited_only)
{
    int ret;

    ret = ibv_req_notify_cq(iface->cq[dir], solicited_only);
    if (ret != 0) {
        ucs_error("ibv_req_notify_cq("UCT_IB_IFACE_FMT", %d, sol=%d) failed: %m",
                  UCT_IB_IFACE_ARG(iface), dir, solicited_only);
        return UCS_ERR_IO_ERROR;
    }
    return UCS_OK;
}

