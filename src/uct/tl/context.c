/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include "context.h"
#include "tl_base.h"

#include <uct/api/uct.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <malloc.h>

#define UCT_CONFIG_ENV_PREFIX "UCT_"

UCS_COMPONENT_LIST_DEFINE(uct_context_t);

/**
 * Keeps information about allocated configuration structure, to be used when
 * releasing the options.
 */
typedef struct uct_config_bundle {
    ucs_config_field_t *table;
    const char         *table_prefix;
    char               data[];
} uct_config_bundle_t;


const char *uct_alloc_method_names[] = {
    [UCT_ALLOC_METHOD_PD]   = "pd",
    [UCT_ALLOC_METHOD_HEAP] = "heap",
    [UCT_ALLOC_METHOD_MMAP] = "mmap",
    [UCT_ALLOC_METHOD_HUGE] = "huge",
    [UCT_ALLOC_METHOD_LAST] = NULL
};


ucs_status_t uct_init(uct_context_h *context_p)
{
    ucs_status_t status;
    uct_context_t *context;

    context = ucs_malloc(ucs_components_total_size(uct_context_t), "uct context");
    if (context == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    context->num_tls  = 0;
    context->tls      = NULL;
    ucs_notifier_chain_init(&context->progress_chain);

    status = ucs_components_init_all(uct_context_t, context);
    if (status != UCS_OK) {
        goto err_free;
    }

    *context_p = context;
    return UCS_OK;

err_free:
    ucs_free(context);
err:
    return status;
}

void uct_cleanup(uct_context_h context)
{
    ucs_free(context->tls);
    ucs_components_cleanup_all(uct_context_t, context);
    ucs_free(context);
}

void uct_progress(uct_context_h context)
{
    ucs_notifier_chain_call(&context->progress_chain);
}

ucs_status_t uct_register_tl(uct_context_h context, const char *tl_name,
                             ucs_config_field_t *config_table, size_t config_size,
                             const char *config_prefix, uct_tl_ops_t *tl_ops)
{
    uct_context_tl_info_t *tls;

    tls = ucs_realloc(context->tls, (context->num_tls + 1) * sizeof(*tls), "tls");
    if (tls == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    context->tls = tls;
    context->tls[context->num_tls].ops                = tl_ops;
    context->tls[context->num_tls].name               = tl_name;
    context->tls[context->num_tls].iface_config_table = config_table;
    context->tls[context->num_tls].iface_config_size  = config_size;
    context->tls[context->num_tls].config_prefix      = config_prefix;
    ++context->num_tls;
    return UCS_OK;
}

ucs_status_t uct_query_resources(uct_context_h context,
                                 uct_resource_desc_t **resources_p,
                                 unsigned *num_resources_p)
{
    uct_resource_desc_t *resources, *tl_resources, *p;
    unsigned num_resources, num_tl_resources, i;
    uct_context_tl_info_t *tl;
    ucs_status_t status;

    resources = NULL;
    num_resources = 0;
    for (tl = context->tls; tl < context->tls + context->num_tls; ++tl) {

        /* Get TL resources */
        status = tl->ops->query_resources(context,&tl_resources,
                                          &num_tl_resources);
        if (status != UCS_OK) {
            continue; /* Skip transport */
        }

        /* tl's query_resources should return error if there are no resources */
        ucs_assert(num_tl_resources > 0);

        /* Enlarge the array */
        p = ucs_realloc(resources, (num_resources + num_tl_resources) * sizeof(*resources),
                        "resources");
        if (p == NULL) {
            goto err_free;
        }
        resources = p;

        /* Append TL resources to the array. Set TL name. */
        for (i = 0; i < num_tl_resources; ++i) {
            resources[num_resources] = tl_resources[i];
            ucs_snprintf_zero(resources[num_resources].tl_name,
                              sizeof(resources[num_resources].tl_name),
                              "%s", tl->name);
            ++num_resources;
        }

        /* Release array returned from TL */
        ucs_free(tl_resources);
    }

    *resources_p     = resources;
    *num_resources_p = num_resources;
    return UCS_OK;

err_free:
    ucs_free(resources);
    return UCS_ERR_NO_MEMORY;
}

void uct_release_resource_list(uct_resource_desc_t *resources)
{
    ucs_free(resources);
}

static uct_context_tl_info_t *uct_find_tl(uct_context_h context, const char *tl_name)
{
    uct_context_tl_info_t *tl;

    for (tl = context->tls; tl < context->tls + context->num_tls; ++tl) {
        if (!strcmp(tl_name, tl->name)) {
            return tl;
        }
    }
    return NULL;
}

ucs_status_t uct_iface_config_read(uct_context_h context, const char *tl_name,
                                   const char *env_prefix, const char *filename,
                                   uct_iface_config_t **config_p)
{
    uct_config_bundle_t *bundle;
    uct_context_tl_info_t *tl;
    ucs_status_t status;

    tl = uct_find_tl(context, tl_name);
    if (tl == NULL) {
        status = UCS_ERR_NO_ELEM; /* Non-existing transport */
        goto err;
    }

    bundle = ucs_calloc(1, sizeof(*bundle) + tl->iface_config_size, "uct_iface_config");
    if (bundle == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* TODO use env_prefix */
    status = ucs_config_parser_fill_opts(bundle->data, tl->iface_config_table,
                                         UCT_CONFIG_ENV_PREFIX, tl->config_prefix,
                                         0);

    if (status != UCS_OK) {
        goto err_free_opts;
    }

    bundle->table        = tl->iface_config_table;
    bundle->table_prefix = tl->config_prefix;
    *config_p = (uct_iface_config_t*)bundle->data; /* coverity[leaked_storage] */
    return UCS_OK;

err_free_opts:
    ucs_free(bundle);
err:
    return status;

}

void uct_iface_config_release(uct_iface_config_t *config)
{
    uct_config_bundle_t *bundle = ucs_container_of(config, uct_config_bundle_t, data);

    ucs_config_parser_release_opts(config, bundle->table);
    ucs_free(bundle);
}

void uct_iface_config_print(uct_iface_config_t *config, FILE *stream,
                            const char *title, ucs_config_print_flags_t print_flags)
{
    uct_config_bundle_t *bundle = ucs_container_of(config, uct_config_bundle_t, data);
    ucs_config_parser_print_opts(stream, title, bundle->data, bundle->table,
                                 UCT_CONFIG_ENV_PREFIX, bundle->table_prefix,
                                 print_flags);
}

ucs_status_t uct_iface_config_modify(uct_iface_config_t *config,
                                     const char *name, const char *value)
{
    uct_config_bundle_t *bundle = ucs_container_of(config, uct_config_bundle_t, data);
    return ucs_config_parser_set_value(bundle->data, bundle->table, name, value);
}

ucs_status_t uct_iface_open(uct_context_h context, const char *tl_name,
                            const char *dev_name, size_t rx_headroom,
                            uct_iface_config_t *config, uct_iface_h *iface_p)
{
    uct_context_tl_info_t *tl = uct_find_tl(context, tl_name);

    if (tl == NULL) {
        /* Non-existing transport */
        return UCS_ERR_NO_DEVICE;
    }

    return tl->ops->iface_open(context, dev_name, rx_headroom, config, iface_p);
}

ucs_status_t uct_rkey_pack(uct_pd_h pd, uct_mem_h memh, void *rkey_buffer)
{
    return pd->ops->rkey_pack(pd, memh, rkey_buffer);
}

ucs_status_t uct_rkey_unpack(uct_context_h context, void *rkey_buffer,
                             uct_rkey_bundle_t *rkey_ob)
{
    uct_context_tl_info_t *tl;
    ucs_status_t status;

    for (tl = context->tls; tl < context->tls + context->num_tls; ++tl) {
        status = tl->ops->rkey_unpack(context, rkey_buffer, rkey_ob);
        if (status != UCS_ERR_UNSUPPORTED) {
            return status;
        }
    }

    return UCS_ERR_INVALID_PARAM;
}

void uct_rkey_release(uct_context_h context, uct_rkey_bundle_t *rkey_ob)
{
    uct_rkey_release_func_t release = rkey_ob->type;
    if (release) {
        release(context, rkey_ob->rkey);
    }
}

ucs_status_t uct_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    return pd->ops->query(pd, pd_attr);
}

ucs_status_t uct_pd_mem_reg(uct_pd_h pd, void *address, size_t length,
                            uct_mem_h *memh_p)
{
    return pd->ops->mem_reg(pd, address, length, memh_p);
}

ucs_status_t uct_pd_mem_dereg(uct_pd_h pd, uct_mem_h memh)
{
    return pd->ops->mem_dereg(pd, memh);
}

static void __uct_free_memory(uct_alloc_method_t method, void *address,
                              size_t length)
{
    int ret;

    switch (method) {
    case UCT_ALLOC_METHOD_HEAP:
        free(address);
        break;

    case UCT_ALLOC_METHOD_MMAP:
        ret = munmap(address, length);
        if (ret != 0) {
            ucs_warn("munmap(address=%p, length=%zu) failed: %m", address, length);
        }
        break;

    case UCT_ALLOC_METHOD_HUGE:
        ucs_sysv_free(address);
        break;

    case UCT_ALLOC_METHOD_PD:
    default:
        break;
    }
}


ucs_status_t uct_pd_mem_alloc(uct_pd_h pd, uct_alloc_method_t method,
                              size_t alignment, const char *alloc_name,
                              size_t *length_p, void **address_p, uct_mem_h *memh_p)
{
    uct_pd_attr_t pd_attr;
    ucs_status_t status;
    uint64_t cap_flag;
    int shmid;

    if ((*length_p == 0) || !ucs_is_pow2(alignment)) {
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_pd_query(pd, &pd_attr);
    if (status != UCS_OK) {
        return status;
    }

    /* Check protection domain capabilities */
    cap_flag = (method == UCT_ALLOC_METHOD_PD) ? UCT_PD_FLAG_ALLOC :
                                                 UCT_PD_FLAG_REG;
    if (!(pd_attr.cap.flags & cap_flag)) {
        return UCS_ERR_UNSUPPORTED;
    }

    *memh_p = UCT_INVALID_MEM_HANDLE;

    switch (method) {
    case UCT_ALLOC_METHOD_PD:
        /* Allocate using protection domain */
        status = pd->ops->mem_alloc(pd, length_p, address_p, memh_p
                                    UCS_MEMTRACK_VAL);
        break;

    case UCT_ALLOC_METHOD_HEAP:
        /* Allocate aligned memory using libc allocator */
        *length_p = ucs_align_up_pow2(*length_p, alignment);
        *address_p = memalign(alignment, *length_p);
        status = (*address_p == NULL) ? UCS_ERR_NO_MEMORY : UCS_OK;
        break;

    case UCT_ALLOC_METHOD_MMAP:
        /* Request memory from operating system using mmap() */
        *length_p = ucs_align_up_pow2(*length_p,
                                      ucs_max(alignment, ucs_get_page_size()));
        *address_p = mmap(NULL, *length_p, PROT_READ|PROT_WRITE,
                          MAP_PRIVATE|MAP_ANON, -1, 0);
        status = (*address_p == MAP_FAILED) ? UCS_ERR_NO_MEMORY : UCS_OK;
        break;

    case UCT_ALLOC_METHOD_HUGE:
        /* Allocate huge pages */
        status = ucs_sysv_alloc(length_p, address_p, SHM_HUGETLB, &shmid);
        break;

    default:
        status = UCS_ERR_INVALID_PARAM;
        break;
    }

    if (status != UCS_OK) {
        return status;
    }

    /* If memory was just allocated, and not registered - register it with the pd */
    if (*memh_p == UCT_INVALID_MEM_HANDLE) {
        status = pd->ops->mem_reg(pd, *address_p, *length_p, memh_p);
        if (status != UCS_OK) {
            __uct_free_memory(method, *address_p, *length_p);
            return status;
        }
    }

    return UCS_OK;
}

ucs_status_t uct_pd_mem_free(uct_pd_h pd, uct_alloc_method_t method, void *address,
                             size_t length, uct_mem_h memh)
{
    if (method == UCT_ALLOC_METHOD_PD) {
        return pd->ops->mem_free(pd, memh);
    } else {
        __uct_free_memory(method, address, length);
        return pd->ops->mem_dereg(pd, memh);
    }
}
