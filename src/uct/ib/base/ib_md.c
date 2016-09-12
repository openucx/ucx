/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * Copyright (C) The University of Tennessee and The University
 *               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ib_md.h"
#include "ib_device.h"

#include <ucs/arch/atomic.h>
#include <pthread.h>


#define UCT_IB_MD_PREFIX         "ib"
#define UCT_IB_MEM_ACCESS_FLAGS  (IBV_ACCESS_LOCAL_WRITE | \
                                  IBV_ACCESS_REMOTE_WRITE | \
                                  IBV_ACCESS_REMOTE_READ | \
                                  IBV_ACCESS_REMOTE_ATOMIC)

static ucs_config_field_t uct_ib_md_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_ib_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

  {"RCACHE", "try", "Enable using memory registration cache",
   ucs_offsetof(uct_ib_md_config_t, rcache.enable), UCS_CONFIG_TYPE_TERNARY},

  {"RCACHE_MEM_PRIO", "1000", "Registration cache memory event priority",
   ucs_offsetof(uct_ib_md_config_t, rcache.event_prio), UCS_CONFIG_TYPE_UINT},

  {"RCACHE_OVERHEAD", "90ns", "Registration cache lookup overhead",
   ucs_offsetof(uct_ib_md_config_t, rcache.overhead), UCS_CONFIG_TYPE_TIME},

  {"MEM_REG_OVERHEAD", "16us", "Memory registration overhead", /* TODO take default from device */
   ucs_offsetof(uct_ib_md_config_t, uc_reg_cost.overhead), UCS_CONFIG_TYPE_TIME},

  {"MEM_REG_GROWTH", "0.06ns", "Memory registration growth rate", /* TODO take default from device */
   ucs_offsetof(uct_ib_md_config_t, uc_reg_cost.growth), UCS_CONFIG_TYPE_TIME},

  {"FORK_INIT", "try",
   "Initialize a fork-safe IB library with ibv_fork_init().",
   ucs_offsetof(uct_ib_md_config_t, fork_init), UCS_CONFIG_TYPE_TERNARY},

  {"ETH_PAUSE_ON", "n",
   "Whether or not 'Pause Frame' is enabled on an Ethernet network.\n"
   "Pause frame is a mechanism for temporarily stopping the transmission of data to\n"
   "ensure zero loss under congestion on Ethernet family computer networks.\n"
   "This parameter, if set to 'no', will disqualify IB transports that may not perform\n"
   "well on a lossy fabric when working with RoCE.",
   ucs_offsetof(uct_ib_md_config_t, eth_pause), UCS_CONFIG_TYPE_BOOL},

  {NULL}
};

#if ENABLE_STATS
static ucs_stats_class_t uct_ib_md_stats_class = {
    .name           = "",
    .num_counters   = UCT_IB_MD_STAT_LAST,
    .counter_names = {
        [UCT_IB_MD_STAT_MEM_ALLOC]   = "mem_alloc",
        [UCT_IB_MD_STAT_MEM_REG]     = "mem_reg"
    }
};
#endif

static ucs_status_t uct_ib_md_query(uct_md_h uct_md, uct_md_attr_t *md_attr)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);

    md_attr->cap.max_alloc = ULONG_MAX; /* TODO query device */
    md_attr->cap.max_reg   = ULONG_MAX; /* TODO query device */
    md_attr->cap.flags     = UCT_MD_FLAG_REG;
    md_attr->rkey_packed_size = sizeof(uint32_t);

    if (IBV_EXP_HAVE_CONTIG_PAGES(&md->dev.dev_attr)) {
        md_attr->cap.flags |= UCT_MD_FLAG_ALLOC;
    }

    md_attr->reg_cost      = md->reg_cost;
    md_attr->local_cpus    = md->dev.local_cpus;
    return UCS_OK;
}

static ucs_status_t uct_ib_dereg_mr(struct ibv_mr *mr)
{
    int ret;

    ret = ibv_dereg_mr(mr);
    if (ret != 0) {
        ucs_error("ibv_dereg_mr() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static ucs_status_t uct_ib_mem_alloc(uct_md_h uct_md, size_t *length_p, void **address_p,
                                     uct_mem_h *memh_p UCS_MEMTRACK_ARG)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    struct ibv_exp_reg_mr_in in = {
        md->pd,
        NULL,
        ucs_memtrack_adjust_alloc_size(*length_p),
        UCT_IB_MEM_ACCESS_FLAGS | IBV_EXP_ACCESS_ALLOCATE_MR,
        0
    };
    struct ibv_mr *mr;

    mr = ibv_exp_reg_mr(&in);
    if (mr == NULL) {
        ucs_error("ibv_exp_reg_mr(in={NULL, length=%Zu, flags=0x%lx}) failed: %m",
                  ucs_memtrack_adjust_alloc_size(*length_p),
                  (unsigned long)(UCT_IB_MEM_ACCESS_FLAGS | IBV_EXP_ACCESS_ALLOCATE_MR));
        return UCS_ERR_IO_ERROR;
    }

    ucs_trace("allocated memory %p..%p on %s lkey 0x%x rkey 0x%x",
              mr->addr, mr->addr + mr->length, uct_ib_device_name(&md->dev),
              mr->lkey, mr->rkey);

    UCS_STATS_UPDATE_COUNTER(md->stats, UCT_IB_MD_STAT_MEM_ALLOC, +1);
    *address_p = mr->addr;
    *length_p  = mr->length;
    ucs_memtrack_allocated(address_p, length_p UCS_MEMTRACK_VAL);
    *memh_p = mr;
    return UCS_OK;
}

static ucs_status_t uct_ib_mem_free(uct_md_h md, uct_mem_h memh)
{
    struct ibv_mr *mr = memh;
    void UCS_V_UNUSED *address = mr->addr;

    ucs_memtrack_releasing(&address);
    return uct_ib_dereg_mr(mr);
}

static ucs_status_t uct_ib_mem_reg(uct_md_h uct_md, void *address, size_t length,
                                   uct_mem_h *memh_p)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    struct ibv_mr *mr;

    mr = ibv_reg_mr(md->pd, address, length, UCT_IB_MEM_ACCESS_FLAGS);
    if (mr == NULL) {
        ucs_error("ibv_reg_mr(address=%p, length=%zu, flags=0x%x) failed: %m",
                  address, length, UCT_IB_MEM_ACCESS_FLAGS);
        return UCS_ERR_IO_ERROR;
    }

    ucs_trace("registered memory %p..%p on %s lkey 0x%x rkey 0x%x",
              address, address + length, uct_ib_device_name(&md->dev), mr->lkey,
              mr->rkey);

    UCS_STATS_UPDATE_COUNTER(md->stats, UCT_IB_MD_STAT_MEM_REG, +1);
    *memh_p = mr;
    return UCS_OK;
}

static ucs_status_t uct_ib_mem_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    struct ibv_mr *mr = memh;
    return uct_ib_dereg_mr(mr);
}

static ucs_status_t uct_ib_mkey_pack(uct_md_h md, uct_mem_h memh,
                                     void *rkey_buffer)
{
    struct ibv_mr *mr = memh;
    *(uint32_t*)rkey_buffer = mr->rkey;
    ucs_trace("packed rkey: 0x%x", mr->rkey);
    return UCS_OK;
}

static ucs_status_t uct_ib_rkey_unpack(uct_md_component_t *mdc,
                                       const void *rkey_buffer, uct_rkey_t *rkey_p,
                                       void **handle_p)
{
    uint32_t ib_rkey = *(const uint32_t*)rkey_buffer;

    *rkey_p   = ib_rkey;
    *handle_p = NULL;
    ucs_trace("unpacked rkey: 0x%x", ib_rkey);
    return UCS_OK;
}

static void uct_ib_md_close(uct_md_h md);

static uct_md_ops_t uct_ib_md_ops = {
    .close        = uct_ib_md_close,
    .query        = uct_ib_md_query,
    .mem_alloc    = uct_ib_mem_alloc,
    .mem_free     = uct_ib_mem_free,
    .mem_reg      = uct_ib_mem_reg,
    .mem_dereg    = uct_ib_mem_dereg,
    .mkey_pack    = uct_ib_mkey_pack,
};

static inline uct_ib_rcache_region_t* uct_ib_rache_region_from_memh(uct_mem_h memh)
{
    return ucs_container_of(memh, uct_ib_rcache_region_t, stub_mr);
}

static void uct_ib_rcache_region_set_stub_mr(uct_ib_rcache_region_t *region)
{
    /* Make sure the lkey and rkey fields of stub_mr do not overlap with 'super' */
    UCS_STATIC_ASSERT(ucs_offsetof(uct_ib_rcache_region_t, stub_mr.lkey) >=
                      ucs_offsetof(uct_ib_rcache_region_t, super) + sizeof(region->super));
    UCS_STATIC_ASSERT(ucs_offsetof(uct_ib_rcache_region_t, stub_mr.rkey) >=
                      ucs_offsetof(uct_ib_rcache_region_t, super) + sizeof(region->super));

    /* Save a copy of lkey and rkey in stub_mr, so transport would be able to use it */
    region->stub_mr.lkey = region->mr->lkey;
    region->stub_mr.rkey = region->mr->rkey;
}

static ucs_status_t
uct_ib_mem_rcache_alloc(uct_md_h uct_md, size_t *length_p, void **address_p,
                        uct_mem_h *memh_p UCS_MEMTRACK_ARG)
{
    uct_ib_rcache_region_t *region;
    ucs_status_t status;

    region = ucs_calloc(1, sizeof(*region), "uct_ib_region");
    if (region == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_ib_mem_alloc(uct_md, length_p, address_p,
                              (uct_mem_h *)&region->mr UCS_MEMTRACK_VAL);
    if (status != UCS_OK) {
        ucs_free(region);
        return status;
    }

    region->super.super.start = (uintptr_t)*address_p;
    region->super.super.end   = (uintptr_t)*address_p + *length_p;

    uct_ib_rcache_region_set_stub_mr(region);
    *memh_p = &region->stub_mr;
    return UCS_OK;
}

static ucs_status_t uct_ib_mem_rcache_free(uct_md_h uct_md, uct_mem_h memh)
{
    uct_ib_rcache_region_t *region = uct_ib_rache_region_from_memh(memh);
    ucs_status_t status;

    status = uct_ib_mem_free(uct_md, region->mr);
    ucs_free(region);
    return status;
}

static ucs_status_t uct_ib_mem_rcache_reg(uct_md_h uct_md, void *address,
                                          size_t length, uct_mem_h *memh_p)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    ucs_rcache_region_t *rregion;
    ucs_status_t status;

    status = ucs_rcache_get(md->rcache, address, length, PROT_READ|PROT_WRITE,
                            &rregion);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(rregion->refcount > 0);
    *memh_p = &ucs_derived_of(rregion, uct_ib_rcache_region_t)->stub_mr;
    return UCS_OK;
}

static ucs_status_t uct_ib_mem_rcache_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    uct_ib_rcache_region_t *region = uct_ib_rache_region_from_memh(memh);

    ucs_rcache_region_put(md->rcache, &region->super);
    return UCS_OK;
}

static uct_md_ops_t uct_ib_md_rcache_ops = {
    .close        = uct_ib_md_close,
    .query        = uct_ib_md_query,
    .mem_alloc    = uct_ib_mem_rcache_alloc,
    .mem_free     = uct_ib_mem_rcache_free,
    .mem_reg      = uct_ib_mem_rcache_reg,
    .mem_dereg    = uct_ib_mem_rcache_dereg,
    .mkey_pack    = uct_ib_mkey_pack,
};


static ucs_status_t uct_ib_rcache_mem_reg_cb(void *context, ucs_rcache_t *rcache,
                                             ucs_rcache_region_t *rregion)
{
    uct_ib_rcache_region_t *region = ucs_derived_of(rregion, uct_ib_rcache_region_t);
    uct_ib_md_t *md = context;
    ucs_status_t status;

    status = uct_ib_mem_reg(&md->super, (void*)region->super.super.start,
                            region->super.super.end - region->super.super.start,
                            (uct_mem_h *)&region->mr);
    if (status != UCS_OK) {
        return status;
    }

    uct_ib_rcache_region_set_stub_mr(region);
    return UCS_OK;
}

static void uct_ib_rcache_mem_dereg_cb(void *context, ucs_rcache_t *rcache,
                                       ucs_rcache_region_t *rregion)
{
    uct_ib_rcache_region_t *region = ucs_derived_of(rregion, uct_ib_rcache_region_t);
    uct_ib_md_t *md = context;

    (void)uct_ib_mem_dereg(&md->super, region->mr);
    region->mr           = NULL;
    region->stub_mr.lkey = 0;
    region->stub_mr.rkey = 0;
}

static void uct_ib_rcache_dump_region_cb(void *context, ucs_rcache_t *rcache,
                                         ucs_rcache_region_t *rregion, char *buf,
                                         size_t max)
{
    uct_ib_rcache_region_t *region = ucs_derived_of(rregion, uct_ib_rcache_region_t);
    snprintf(buf, max, "lkey 0x%x rkey 0x%x", region->mr->lkey, region->mr->rkey);
}

static ucs_rcache_ops_t uct_ib_rcache_ops = {
    .mem_reg     = uct_ib_rcache_mem_reg_cb,
    .mem_dereg   = uct_ib_rcache_mem_dereg_cb,
    .dump_region = uct_ib_rcache_dump_region_cb
};

static void uct_ib_make_md_name(char md_name[UCT_MD_NAME_MAX], struct ibv_device *device)
{
    snprintf(md_name, UCT_MD_NAME_MAX, "%s/%s", UCT_IB_MD_PREFIX, device->name);
}

static ucs_status_t uct_ib_query_md_resources(uct_md_resource_desc_t **resources_p,
                                              unsigned *num_resources_p)
{
    uct_md_resource_desc_t *resources;
    struct ibv_device **device_list;
    ucs_status_t status;
    int i, num_devices;

    /* Get device list from driver */
    device_list = ibv_get_device_list(&num_devices);
    if (device_list == NULL) {
        ucs_debug("Failed to get IB device list, assuming no devices are present");
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    resources = ucs_calloc(num_devices, sizeof(*resources), "ib resources");
    if (resources == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_free_device_list;
    }

    for (i = 0; i < num_devices; ++i) {
        uct_ib_make_md_name(resources[i].md_name, device_list[i]);
    }

    *resources_p     = resources;
    *num_resources_p = num_devices;
    status = UCS_OK;

out_free_device_list:
    ibv_free_device_list(device_list);
out:
    return status;
}

static void uct_ib_fork_warn()
{
    ucs_warn("ibv_fork_init() was not successful, yet a fork() has been issued.");
}

static void uct_ib_fork_warn_enable()
{
    static volatile uint32_t enabled = 0;
    int ret;

    if (ucs_atomic_cswap32(&enabled, 0, 1) != 0) {
        return;
    }

    ret = pthread_atfork(uct_ib_fork_warn, NULL, NULL);
    if (ret) {
        ucs_warn("ibv_fork_init failed, and registering atfork warning failed too: %m");
    }
}

static ucs_status_t
uct_ib_md_open(const char *md_name, const uct_md_config_t *uct_md_config, uct_md_h *md_p)
{
    const uct_ib_md_config_t *md_config = ucs_derived_of(uct_md_config, uct_ib_md_config_t);
    struct ibv_device **ib_device_list, *ib_device;
    char tmp_md_name[UCT_MD_NAME_MAX];
    ucs_rcache_params_t rcache_params;
    ucs_status_t status;
    int i, num_devices, ret;
    uct_ib_md_t *md;

    /* Get device list from driver */
    ib_device_list = ibv_get_device_list(&num_devices);
    if (ib_device_list == NULL) {
        ucs_debug("Failed to get IB device list, assuming no devices are present");
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    ib_device = NULL;
    for (i = 0; i < num_devices; ++i) {
        uct_ib_make_md_name(tmp_md_name, ib_device_list[i]);
        if (!strcmp(tmp_md_name, md_name)) {
            ib_device = ib_device_list[i];
            break;
        }
    }
    if (ib_device == NULL) {
        status = UCS_ERR_NO_DEVICE;
        goto out_free_dev_list;
    }

    md = ucs_malloc(sizeof(*md), "ib_md");
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_free_dev_list;
    }

    md->super.ops       = &uct_ib_md_ops;
    md->super.component = &uct_ib_mdc;

    /* Create statistics */
    status = UCS_STATS_NODE_ALLOC(&md->stats, &uct_ib_md_stats_class, NULL,
                                  "%s-%p", ibv_get_device_name(ib_device), md);
    if (status != UCS_OK) {
        goto err_free_md;
    }

    if (md_config->fork_init != UCS_NO) {
        ret = ibv_fork_init();
        if (ret) {
            if (md_config->fork_init == UCS_YES) {
                ucs_error("ibv_fork_init() failed: %m");
                status = UCS_ERR_IO_ERROR;
                goto err_release_stats;
            }
            ucs_debug("ibv_fork_init() failed: %m, continuing, but fork may be unsafe.");
            uct_ib_fork_warn_enable();
        }
    }

    status = uct_ib_device_init(&md->dev, ib_device UCS_STATS_ARG(md->stats));
    if (status != UCS_OK) {
        goto err_release_stats;
    }

    /* Allocate memory domain */
    md->pd = ibv_alloc_pd(md->dev.ibv_context);
    if (md->pd == NULL) {
        ucs_error("ibv_alloc_pd() failed: %m");
        status = UCS_ERR_NO_MEMORY;
        goto err_cleanup_device;
    }

    md->eth_pause = md_config->eth_pause;

    md->rcache   = NULL;
    md->reg_cost = md_config->uc_reg_cost;

    if (md_config->rcache.enable != UCS_NO) {
        rcache_params.region_struct_size = sizeof(uct_ib_rcache_region_t);
        rcache_params.ucm_event_priority = md_config->rcache.event_prio;
        rcache_params.context            = md;
        rcache_params.ops                = &uct_ib_rcache_ops;
        status = ucs_rcache_create(&rcache_params, uct_ib_device_name(&md->dev)
                                   UCS_STATS_ARG(md->stats), &md->rcache);
        if (status == UCS_OK) {
            md->super.ops         = &uct_ib_md_rcache_ops;
            md->reg_cost.overhead = md_config->rcache.overhead;
            md->reg_cost.growth   = 0; /* It's close enough to 0 */
        } else {
            ucs_assert(md->rcache == NULL);
            if (md_config->rcache.enable == UCS_YES) {
                ucs_error("Failed to create registration cache: %s",
                          ucs_status_string(status));
                goto err_dealloc_pd;
            } else {
                ucs_debug("Could not create registration cache for: %s",
                          ucs_status_string(status));
            }
        }
    }

    *md_p = &md->super;
    status = UCS_OK;

out_free_dev_list:
    ibv_free_device_list(ib_device_list);
out:
    return status;

err_dealloc_pd:
    ibv_dealloc_pd(md->pd);
err_cleanup_device:
    uct_ib_device_cleanup(&md->dev);
err_release_stats:
    UCS_STATS_NODE_FREE(md->stats);
err_free_md:
    ucs_free(md);
    goto out_free_dev_list;
}

static void uct_ib_md_close(uct_md_h uct_md)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);

    if (md->rcache != NULL) {
        ucs_rcache_destroy(md->rcache);
    }
    ibv_dealloc_pd(md->pd);
    uct_ib_device_cleanup(&md->dev);
    UCS_STATS_NODE_FREE(md->stats);
    ucs_free(md);
}

UCT_MD_COMPONENT_DEFINE(uct_ib_mdc, UCT_IB_MD_PREFIX,
                        uct_ib_query_md_resources, uct_ib_md_open, NULL,
                        uct_ib_rkey_unpack,
                        (void*)ucs_empty_function_return_success /* release */,
                        "IB_", uct_ib_md_config_table, uct_ib_md_config_t);
