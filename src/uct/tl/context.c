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
#include <ucs/type/class.h>
#include <malloc.h>

#define UCT_CONFIG_ENV_PREFIX "UCT_"

UCS_COMPONENT_LIST_DEFINE(uct_context_t);
UCS_LIST_HEAD(uct_pd_components_list);

/**
 * Keeps information about allocated configuration structure, to be used when
 * releasing the options.
 */
typedef struct uct_config_bundle {
    ucs_config_field_t *table;
    const char         *table_prefix;
    char               data[];
} uct_config_bundle_t;


/**
 * Header of allocated memory block. Stores information used later to release
 * the memory.
 *
 * +---------+--------+-----------------+
 * | padding | header | user memory ... |
 * +---------+--------+-----------------+
 */
typedef struct uct_mem_block_header {
    uct_alloc_method_t alloc_method;  /**< Method used to allocate the memory */
    size_t             total_length;  /**< Total allocation length */
    void               *mem_block;    /**< Points to the beginning of the whole block */
} uct_mem_block_header_t;


const char *uct_alloc_method_names[] = {
    [UCT_ALLOC_METHOD_PD]   = "pd",
    [UCT_ALLOC_METHOD_HEAP] = "heap",
    [UCT_ALLOC_METHOD_MMAP] = "mmap",
    [UCT_ALLOC_METHOD_HUGE] = "huge",
    [UCT_ALLOC_METHOD_LAST] = NULL
};


ucs_status_t uct_query_pd_resources(uct_pd_resource_desc_t **resources_p,
                                    unsigned *num_resources_p)
{
    uct_pd_resource_desc_t *resources, *pd_resources, *tmp;
    unsigned i, num_resources, num_pd_resources;
    uct_pd_component_t *pdc;
    ucs_status_t status;

    resources     = NULL;
    num_resources = 0;

    ucs_list_for_each(pdc, &uct_pd_components_list, list) {
        status = pdc->query_resources(&pd_resources, &num_pd_resources);
        if (status != UCS_OK) {
            ucs_debug("Failed to query %s* resources: %s", pdc->name_prefix,
                      ucs_status_string(status));
            continue;
        }

        tmp = ucs_realloc(resources,
                          sizeof(*resources) * (num_resources + num_pd_resources),
                          "pd_resources");
        if (tmp == NULL) {
            ucs_free(pd_resources);
            status = UCS_ERR_NO_MEMORY;
            goto err;
        }

        for (i = 0; i < num_pd_resources; ++i) {
            ucs_assert_always(!strncmp(pdc->name_prefix, pd_resources[i].pd_name,
                                       strlen(pdc->name_prefix)));
        }
        resources = tmp;
        memcpy(resources + num_resources, pd_resources,
               sizeof(*pd_resources) * num_pd_resources);
        num_resources += num_pd_resources;
        ucs_free(pd_resources);
    }

    *resources_p     = resources;
    *num_resources_p = num_resources;
    return UCS_OK;

err:
    ucs_free(resources);
    return status;
}

void uct_release_pd_resource_list(uct_pd_resource_desc_t *resources)
{
    ucs_free(resources);
}

ucs_status_t uct_pd_open(const char *pd_name, uct_pd_h *pd_p)
{
    uct_pd_component_t *pdc;
    ucs_status_t status;
    uct_pd_h pd;

    ucs_list_for_each(pdc, &uct_pd_components_list, list) {
        if (!strncmp(pd_name, pdc->name_prefix, strlen(pdc->name_prefix))) {
            status = pdc->pd_open(pd_name, &pd);
            if (status != UCS_OK) {
                return status;
            }

            ucs_assert_always(pd->component == pdc);
            *pd_p = pd;
            return UCS_OK;
        }
    }

    ucs_error("PD '%s' does not exist", pd_name);
    return UCS_ERR_NO_DEVICE;
}

void uct_pd_close(uct_pd_h pd)
{
    pd->ops->close(pd);
}

ucs_status_t uct_pd_query_tl_resources(uct_pd_h pd,
                                       uct_tl_resource_desc_t **resources_p,
                                       unsigned *num_resources_p)
{
    uct_tl_resource_desc_t *resources, *tl_resources, *tmp;
    unsigned i, num_resources, num_tl_resources;
    uct_pd_component_t *pdc = pd->component;
    uct_tl_component_t *tlc;
    ucs_status_t status;

    resources     = NULL;
    num_resources = 0;

    ucs_list_for_each(tlc, &pdc->tl_list, list) {
        status = tlc->query_resources(pd, &tl_resources, &num_tl_resources);
        if (status != UCS_OK) {
            ucs_debug("Failed to query %s resources: %s", tlc->name,
                      ucs_status_string(status));
            continue;
        }

        tmp = ucs_realloc(resources,
                          sizeof(*resources) * (num_resources + num_tl_resources),
                          "pd_resources");
        if (tmp == NULL) {
            ucs_free(tl_resources);
            status = UCS_ERR_NO_MEMORY;
            goto err;
        }

        for (i = 0; i < num_tl_resources; ++i) {
            ucs_assert_always(!strcmp(tlc->name, tl_resources[i].tl_name));
        }
        resources = tmp;
        memcpy(resources + num_resources, tl_resources,
               sizeof(*tl_resources) * num_tl_resources);
        num_resources += num_tl_resources;
        ucs_free(tl_resources);
    }

    *resources_p     = resources;
    *num_resources_p = num_resources;
    return UCS_OK;

err:
    ucs_free(resources);
    return status;
}

void uct_release_tl_resource_list(uct_tl_resource_desc_t *resources)
{
    ucs_free(resources);
}


ucs_status_t uct_single_pd_resource(uct_pd_component_t *pdc,
                                    uct_pd_resource_desc_t **resources_p,
                                    unsigned *num_resources_p)
{
    uct_pd_resource_desc_t *resource;

    resource = ucs_malloc(sizeof(*resource), "pd resource");
    if (resource == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    ucs_snprintf_zero(resource->pd_name, UCT_PD_NAME_MAX, "%s", pdc->name_prefix);

    *resources_p     = resource;
    *num_resources_p = 1;
    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_worker_t, ucs_async_context_t *async,
                           ucs_thread_mode_t thread_mode)
{
    self->async       = async;
    self->thread_mode = thread_mode;
    ucs_notifier_chain_init(&self->progress_chain);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_worker_t)
{
    /* TODO warn if notifier chain is non-empty */
}

void uct_worker_progress(uct_worker_h worker)
{
    ucs_notifier_chain_call(&worker->progress_chain);
}

UCS_CLASS_DEFINE(uct_worker_t, void);
UCS_CLASS_DEFINE_NAMED_NEW_FUNC(uct_worker_create, uct_worker_t, uct_worker_t,
                                ucs_async_context_t*, ucs_thread_mode_t)
UCS_CLASS_DEFINE_NAMED_DELETE_FUNC(uct_worker_destroy, uct_worker_t, uct_worker_t)


static uct_tl_component_t *uct_find_tl_on_pd(uct_pd_component_t *pdc,
                                             const char *tl_name)
{
    uct_tl_component_t *tlc;

    ucs_list_for_each(tlc, &pdc->tl_list, list) {
        if (!strcmp(tl_name, tlc->name)) {
            return tlc;
        }
    }
    return NULL;
}

static uct_tl_component_t *uct_find_tl(const char *tl_name)
{
    uct_pd_component_t *pdc;
    uct_tl_component_t *tlc;

    ucs_list_for_each(pdc, &uct_pd_components_list, list) {
        tlc = uct_find_tl_on_pd(pdc, tl_name);
        if (tlc != NULL) {
            return tlc;
        }
    }
    return NULL;
}

ucs_status_t uct_iface_config_read(const char *tl_name, const char *env_prefix,
                                   const char *filename,
                                   uct_iface_config_t **config_p)
{
    uct_config_bundle_t *bundle;
    uct_tl_component_t *tlc;
    ucs_status_t status;

    tlc = uct_find_tl(tl_name);
    if (tlc == NULL) {
        ucs_error("Transport '%s' does not exist", tl_name);
        status = UCS_ERR_NO_DEVICE; /* Non-existing transport */
        goto err;
    }

    bundle = ucs_calloc(1, sizeof(*bundle) + tlc->iface_config_size,
                        "uct_iface_config");
    if (bundle == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* TODO use env_prefix */
    status = ucs_config_parser_fill_opts(bundle->data, tlc->iface_config_table,
                                         UCT_CONFIG_ENV_PREFIX, tlc->cfg_prefix,
                                         0);

    if (status != UCS_OK) {
        goto err_free_opts;
    }

    bundle->table        = tlc->iface_config_table;
    bundle->table_prefix = tlc->cfg_prefix;
    *config_p = (uct_iface_config_t*)bundle->data; /* coverity[leaked_storage] */
    return UCS_OK;

err_free_opts:
    ucs_free(bundle);
err:
    return status;

}

void uct_iface_config_release(uct_iface_config_t *config)
{
    uct_config_bundle_t *bundle = ucs_container_of(config, uct_config_bundle_t,
                                                   data);

    ucs_config_parser_release_opts(config, bundle->table);
    ucs_free(bundle);
}

void uct_iface_config_print(const uct_iface_config_t *config, FILE *stream,
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

ucs_status_t uct_iface_open(uct_pd_h pd, uct_worker_h worker, const char *tl_name,
                            const char *dev_name, size_t rx_headroom,
                            const uct_iface_config_t *config, uct_iface_h *iface_p)
{
    uct_tl_component_t *tlc;

    tlc = uct_find_tl_on_pd(pd->component, tl_name);
    if (tlc == NULL) {
        /* Non-existing transport */
        return UCS_ERR_NO_DEVICE;
    }

    return tlc->iface_open(pd, worker, dev_name, rx_headroom, config, iface_p);
}

ucs_status_t uct_pd_rkey_pack(uct_pd_h pd, uct_mem_h memh, void *rkey_buffer)
{
    return pd->ops->rkey_pack(pd, memh, rkey_buffer);
}

ucs_status_t uct_pd_rkey_unpack(uct_pd_h pd, const void *rkey_buffer,
                                uct_rkey_bundle_t *rkey_ob)
{
    return pd->ops->rkey_unpack(pd, rkey_buffer, rkey_ob);
}

void uct_pd_rkey_release(uct_pd_h pd, const uct_rkey_bundle_t *rkey_ob)
{
    pd->ops->rkey_release(pd, rkey_ob);
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
        ucs_free(address);
        break;

    case UCT_ALLOC_METHOD_MMAP:
        ret = ucs_munmap(address, length);
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

static ucs_status_t uct_pd_mem_alloc_method(uct_pd_h pd, uct_pd_attr_t *pd_attr,
                                            uct_alloc_method_t method, size_t *length_p,
                                            size_t alignment, void **address_p,
                                            uct_mem_h *memh_p, const char *alloc_name)
{
    uct_mem_block_header_t *header;
    ucs_status_t status;
    size_t alloc_length;
    void *block;
    int shmid;

    ucs_assert(alignment >= 1);
    alloc_length = *length_p + alignment - 1 + sizeof(*header);

    if (method == UCT_ALLOC_METHOD_PD) {
        if (!(pd_attr->cap.flags & UCT_PD_FLAG_ALLOC)) {
            ucs_debug("pd does not support allocation");
            return UCS_ERR_UNSUPPORTED;
        }

        /* Allocate using protection domain */
        status = pd->ops->mem_alloc(pd, &alloc_length, &block, memh_p
                                    UCS_MEMTRACK_VAL);
        if (status != UCS_OK) {
            ucs_debug("failed to allocate memory using pd");
            return status;
        }

    } else {
        if (!(pd_attr->cap.flags & UCT_PD_FLAG_REG)) {
            ucs_debug("pd does not support registration");
            return UCS_ERR_UNSUPPORTED;
        }

        switch (method) {
        case UCT_ALLOC_METHOD_HEAP:
            /* Allocate aligned memory using libc allocator */
            block = ucs_malloc(alloc_length UCS_MEMTRACK_VAL);
            status = (block == NULL) ? UCS_ERR_NO_MEMORY : UCS_OK;
            break;

        case UCT_ALLOC_METHOD_MMAP:
            /* Request memory from operating system using mmap() */
            alloc_length = ucs_align_up_pow2(alloc_length, ucs_get_page_size());
            block = ucs_mmap(NULL, alloc_length, PROT_READ|PROT_WRITE,
                             MAP_PRIVATE|MAP_ANON, -1, 0 UCS_MEMTRACK_VAL);
            status = (block == MAP_FAILED) ? UCS_ERR_NO_MEMORY : UCS_OK;
            break;

        case UCT_ALLOC_METHOD_HUGE:
            /* Allocate huge pages */
            status = ucs_sysv_alloc(&alloc_length, &block, SHM_HUGETLB, &shmid
                                    UCS_MEMTRACK_VAL);
            break;

        default:
            ucs_error("Invalid allocation method %d", method);
            status = UCS_ERR_INVALID_PARAM;
            break;
        }

        if (status != UCS_OK) {
            ucs_debug("failed to allocate %zu bytes using %s", alloc_length,
                      uct_alloc_method_names[method]);
            return status;
        }

        /* Register memory on PD */
        status = pd->ops->mem_reg(pd, block, alloc_length, memh_p);
        if (status != UCS_OK) {
            ucs_debug("failed to register memory");
            __uct_free_memory(method, block, alloc_length);
            return status;
        }
    }

    /* Align the address returned to the user. */
    *address_p = (void*)ucs_align_up_pow2((uintptr_t)(block + sizeof(*header)),
                                          alignment);

    /* Adjust the length. */
    ucs_assert(block + alloc_length - *address_p >= *length_p);
    *length_p  = block + alloc_length - *address_p;

    /* Save memory block information in the header.
     * (The header lies just before the pointer returned to the user).
     */
    header               = *address_p - sizeof(*header);
    header->alloc_method = method;
    header->total_length = alloc_length;
    header->mem_block    = block;

    VALGRIND_MAKE_MEM_NOACCESS(block, *address_p - block);
    ucs_debug("allocated %zu (%zu) bytes using %s: %p (%p)", *length_p, alloc_length,
              uct_alloc_method_names[method], *address_p, block);
    return UCS_OK;
}

ucs_status_t uct_pd_mem_alloc(uct_pd_h pd, uct_alloc_method_t method,
                              size_t *length_p, size_t alignment, void **address_p,
                              uct_mem_h *memh_p, const char *alloc_name)
{
    uct_pd_attr_t pd_attr;
    ucs_status_t status;
    uint8_t i;

    if (*length_p == 0) {
        ucs_error("Allocation length cannot be 0");
        return UCS_ERR_INVALID_PARAM;
    }

    if (!(alignment >= 1) || !ucs_is_pow2(alignment)) {
        ucs_error("Allocation alignment must be power of 2 (got: %zu)", alignment);
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_pd_query(pd, &pd_attr);
    if (status != UCS_OK) {
        ucs_debug("failed to query pd");
        return status;
    }

    if (method != UCT_ALLOC_METHOD_DEFAULT) {
        /* Allocate using specific method */
        status = uct_pd_mem_alloc_method(pd, &pd_attr, method, length_p,
                                         alignment, address_p, memh_p, alloc_name);
    } else {
        /* Allocate using default method */
        for (i = 0; i < pd_attr.alloc_methods.count; ++i) {
            status = uct_pd_mem_alloc_method(pd, &pd_attr,
                                             pd_attr.alloc_methods.methods[i],
                                             length_p, alignment, address_p,
                                             memh_p, alloc_name);
            if (status == UCS_OK) {
                return UCS_OK;
            }
        }
        status = UCS_ERR_NO_MEMORY; /* No more methods to try */
    }

    if (status != UCS_OK) {
        ucs_debug("could not allocate memory");
    }
    return status;
}

ucs_status_t uct_pd_mem_free(uct_pd_h pd, void *address, uct_mem_h memh)
{
    uct_mem_block_header_t *header = address - sizeof(uct_mem_block_header_t);

    VALGRIND_MAKE_MEM_DEFINED(header, sizeof(*header));

    if (header->alloc_method == UCT_ALLOC_METHOD_PD) {
        return pd->ops->mem_free(pd, memh);
    } else {
        __uct_free_memory(header->alloc_method, header->mem_block,
                          header->total_length);
        return pd->ops->mem_dereg(pd, memh);
    }
}
