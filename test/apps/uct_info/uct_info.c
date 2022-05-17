/**
* Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <alloca.h>
#include <string.h>

#include <uct/api/uct.h>
#include <ucs/sys/string.h>
#include <ucs/sys/compiler_def.h>


#define CALL(_action, _err) \
    do { \
        ucs_status_t UCS_V_UNUSED _status; \
        _status = _action; \
        if (_status != UCS_OK) { \
            _err; \
        } \
    } while (0)


#define UCT_CALL(_action, _err) \
    CALL(_action, { \
        printf("ERROR: %s failed\n", #_action); \
        _err; \
    })


static ucs_status_t uct_info_iface_info(uct_worker_h worker, uct_md_h md,
                                        uct_tl_resource_desc_t *resource)
{
    uct_iface_params_t iface_params = {
        .field_mask           = UCT_IFACE_PARAM_FIELD_OPEN_MODE |
                                UCT_IFACE_PARAM_FIELD_DEVICE,
        .open_mode            = UCT_IFACE_OPEN_MODE_DEVICE,
        .mode.device.tl_name  = resource->tl_name,
        .mode.device.dev_name = resource->dev_name
    };
    ucs_status_t status             = UCS_OK;
    uct_iface_config_t *iface_config;
    uct_iface_h iface;

    UCT_CALL(uct_md_iface_config_read(md, resource->tl_name, NULL, NULL,
                                      &iface_config),
             return _status);

    printf("#      Transport: %s\n", resource->tl_name);
    printf("#         Device: %s\n", resource->dev_name);
    printf("#           Type: %s\n", uct_device_type_names[resource->dev_type]);
    printf("#  System device: %s\n",
           ucs_topo_sys_device_get_name(resource->sys_device));

    UCT_CALL(uct_iface_open(md, worker, &iface_params, iface_config, &iface), {
        status = _status;
        goto out;
    });

    uct_iface_close(iface);
out:
    uct_config_release(iface_config);
    printf("#\n");
    return status;
}

static ucs_status_t uct_info_tl_info(uct_md_h md,
                                     uct_tl_resource_desc_t *resources,
                                     unsigned num_resources)
{
    ucs_status_t status;
    ucs_async_context_t *async;
    uct_worker_h worker;
    unsigned i;

    UCT_CALL(ucs_async_context_create(UCS_ASYNC_MODE_THREAD_SPINLOCK, &async),
             return _status);
    UCT_CALL(uct_worker_create(async, UCS_THREAD_MODE_SINGLE, &worker), {
        status = _status;
        goto destroy_async;
    });

    printf("#\n");

    if (num_resources == 0) {
        printf("# (No supported devices found)\n");
    }

    for (i = 0; i < num_resources; ++i) {
        CALL(uct_info_iface_info(worker, md, &resources[i]), break);
    }

    uct_worker_destroy(worker);
    status = UCS_OK;

destroy_async:
    ucs_async_context_destroy(async);
    return status;
}

static ucs_status_t uct_info_md_info(uct_component_h component,
                                     const uct_component_attr_t *component_attr,
                                     const char *md_name)
{
    ucs_status_t status;
    uct_tl_resource_desc_t *resources;
    unsigned num_resources;
    uct_md_config_t *md_config;
    uct_md_attr_t md_attr;
    uct_md_h md;

    UCT_CALL(uct_md_config_read(component, NULL, NULL, &md_config),
             return _status);
    UCT_CALL(uct_md_open(component, md_name, md_config, &md), {
        status = _status;
        goto out_release_config;
    });
    UCT_CALL(uct_md_query_tl_resources(md, &resources, &num_resources), {
        status = _status;
        goto out_close_md;
    });
    UCT_CALL(uct_md_query(md, &md_attr), {
        status = _status;
        goto out_release_resources;
    });

    printf("#\n");
    printf("# Memory domain: %s\n", md_name);
    printf("#     Component: %s\n", component_attr->name);

    if (num_resources == 0) {
        printf("#   < no supported devices found >\n");
        goto out_release_resources;
    }

    CALL(uct_info_tl_info(md, resources, num_resources), break);
    status = UCS_OK;

out_release_resources:
    uct_release_tl_resource_list(resources);
out_close_md:
    uct_md_close(md);
out_release_config:
    uct_config_release(md_config);

    return status;
}

static ucs_status_t uct_info_component(uct_component_h component)
{
    uct_component_attr_t component_attr;
    unsigned i;

    component_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_NAME |
                                UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT |
                                UCT_COMPONENT_ATTR_FIELD_FLAGS;
    UCT_CALL(uct_component_query(component, &component_attr), return _status);

    component_attr.field_mask   = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
    component_attr.md_resources = alloca(sizeof(*component_attr.md_resources) *
                                         component_attr.md_resource_count);
    UCT_CALL(uct_component_query(component, &component_attr), return _status);

    for (i = 0; i < component_attr.md_resource_count; ++i) {
        CALL(uct_info_md_info(component, &component_attr,
                              component_attr.md_resources[i].md_name),
             return _status);
    }

    return UCS_OK;
}

int main(int argc, char **argv)
{
    int res;
    uct_component_h *components;
    unsigned num_components;
    unsigned i;

    UCT_CALL(uct_query_components(&components, &num_components),
             return EXIT_FAILURE);

    for (i = 0; i < num_components; i++) {
        CALL(uct_info_component(components[i]), {
            res = EXIT_FAILURE;
            goto out;
        });
    }

    res = EXIT_SUCCESS;

out:
    uct_release_component_list(components);
    return res;
}
