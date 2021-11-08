
#include <uct/base/uct_md.h>
#include <ucs/type/class.h>
#include <ucs/type/status.h>
//#include <uct/tcp/tcp_md.c>
#include "stdio.h"
#include "sisci.h"


/* Forward declarations */
static uct_iface_ops_t uct_sisci_iface_ops;




static ucs_config_field_t uct_sisci_iface_config_table[] = {
    NULL
};


static ucs_config_field_t uct_sisci_md_config_table[] = {
    NULL
};



//various "class" funcitons, don't really know how they work yet, but seems to be some sort of glue code. 
//also known as "macro hell"
static UCS_CLASS_CLEANUP_FUNC(uct_sisci_ep_t)
{
}

static UCS_CLASS_INIT_FUNC(uct_sisci_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    printf("UCS_CLASS_INIT_FUNC\n");
    UCS_CLASS_CALL_SUPER_INIT(
            uct_base_iface_t, &uct_sisci_iface_ops,
            &uct_base_iface_internal_ops, md, worker, params,
            tl_config UCS_STATS_ARG(
                    (params->field_mask & UCT_IFACE_PARAM_FIELD_STATS_ROOT) ?
                            params->stats_root :
                            NULL) UCS_STATS_ARG(UCT_sisci_NAME));
    
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_sisci_iface_t)
{
    ucs_mpool_cleanup(&self->msg_mp, 1);
}

UCS_CLASS_DEFINE(uct_sisci_iface_t, uct_base_iface_t);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_sisci_iface_t, uct_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_sisci_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);



static UCS_CLASS_INIT_FUNC(uct_sisci_ep_t, const uct_ep_params_t *params)
{
    //uct_self_iface_t *iface = ucs_derived_of(params->iface, uct_self_iface_t);

    //UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super)
    

    return UCS_OK;
}

UCS_CLASS_DEFINE(uct_sisci_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_sisci_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_sisci_ep_t, uct_ep_t);




static ucs_status_t uct_sisci_query_md_resources(uct_component_t *component,
                                              uct_md_resource_desc_t **resources_p,
                                              unsigned *num_resources_p)
{
    printf("SISCI: UCT_SICI_QUERY_MD_RESOURCES\n");
    return UCS_OK;

}


static ucs_status_t uct_sisci_query_devices(uct_md_h md,
                                   uct_tl_device_resource_t **devices_p,
                                   unsigned *num_devices_p)
{

    printf("UCT_SISCI_QUERY_DEVICES\n");
    return UCS_OK;
}


static ucs_status_t uct_sisci_md_open(uct_component_t *component, const char *md_name,
                                     const uct_md_config_t *config, uct_md_h *md_p)
{
    printf("UCT_SISCI_MD_OPEN\n");
    return UCS_OK;
}


static ucs_status_t uct_sisci_md_rkey_unpack(uct_component_t *component,
                                            const void *rkey_buffer, uct_rkey_t *rkey_p,
                                            void **handle_p)
{
    /**
     * Pseudo stub function for the key unpacking
     * Need rkey == 0 due to work with same process to reuse uct_base_[put|get|atomic]*
     */
    *rkey_p   = 0;
    *handle_p = NULL;
    return UCS_OK;
}

/*
    TODO: Figure out what to change the commented lines to : )
*/
uct_component_t uct_sisci_component = {
    .query_md_resources = uct_sisci_query_md_resources, 
    .md_open            = uct_sisci_md_open,
    .cm_open            = ucs_empty_function_return_unsupported, //UCS_CLASS_NEW_FUNC_NAME(uct_tcp_sockcm_t), //change me
    .rkey_unpack        = uct_sisci_md_rkey_unpack, //change me
    .rkey_ptr           = ucs_empty_function_return_unsupported, //change me 
    .rkey_release       = ucs_empty_function_return_success, //change me
    .name               = UCT_SISCI_NAME, //change me
    .md_config          = {
        .name           = "Self memory domain",
        .prefix         = "SELF_",
        .table          = uct_sisci_md_config_table,
        .size           = sizeof(uct_sisci_md_config_t),
    },
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_sisci_component),
    .flags              = 0, //UCT_COMPONENT_FLAG_CM,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
UCT_COMPONENT_REGISTER(&uct_sisci_component)


//the operations that we should support or something : )
static uct_iface_ops_t uct_sisci_iface_ops = {
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_sisci_ep_t),

    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_sisci_iface_t),

};


/*
    TODO: Add the mimimum stuff required to get it to compile.
*/
UCT_TL_DEFINE(&uct_sisci_component, sisci, uct_sisci_query_devices, uct_sisci_iface_t,
              UCT_SISCI_CONFIG_PREFIX, uct_sisci_iface_config_table, uct_sisci_iface_config_t);


/* 
static uct_component_t uct_self_component = {
    .query_md_resources = uct_md_query_single_md_resource,
    .md_open            = uct_self_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_self_md_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = ucs_empty_function_return_success,
    .name               = UCT_SELF_NAME,
    .md_config          = {
        .name           = "Self memory domain",
        .prefix         = "SELF_",
        .table          = uct_self_md_config_table,
        .size           = sizeof(uct_self_md_config_t),
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_self_component),
    .flags              = 0,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
UCT_COMPONENT_REGISTER(&uct_self_component);



UCT_TL_DEFINE(&uct_self_component, self, uct_self_query_tl_devices, uct_self_iface_t,
              "SELF_", uct_self_iface_config_table, uct_self_iface_config_t);
*/


