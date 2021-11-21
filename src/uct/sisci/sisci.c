
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

void sisci_testing() {
    printf("Linking is correct to some degree :) \n");
}

//various "class" funcitons, don't really know how they work yet, but seems to be some sort of glue code. 
//also known as "macro hell"
static UCS_CLASS_CLEANUP_FUNC(uct_sisci_ep_t)
{
}

static UCS_CLASS_INIT_FUNC(uct_sisci_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    printf("UCS_SISCI_CLASS_INIT_FUNC\n");
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


    return UCS_ERR_NOT_IMPLEMENTED;
}

UCS_CLASS_DEFINE(uct_sisci_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_sisci_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_sisci_ep_t, uct_ep_t);




static ucs_status_t uct_sisci_query_md_resources(uct_component_t *component,
                                              uct_md_resource_desc_t **resources_p,
                                              unsigned int *num_resources_p)
{
    printf("SISCI: UCT_SICI_QUERY_MD_RESOURCES\n");
    return UCS_ERR_NOT_IMPLEMENTED;

}


static ucs_status_t uct_sisci_query_devices(uct_md_h md,
                                   uct_tl_device_resource_t **devices_p,
                                   unsigned *num_devices_p)
{

    printf("UCT_SISCI_QUERY_DEVICES\n");
    return UCS_OK;
}




static ucs_status_t uct_sisci_md_query(uct_md_h md, uct_md_attr_t *attr)
{
    /* Dummy memory registration provided. No real memory handling exists */
    
    
    attr->cap.flags            = UCT_MD_FLAG_REG |
                                 UCT_MD_FLAG_NEED_RKEY; // TODO ignore rkey in rma/amo ops 
    attr->cap.reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    attr->cap.detect_mem_types = 0;
    attr->cap.access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    attr->cap.max_alloc        = 0;
    attr->cap.max_reg          = ULONG_MAX;
    attr->rkey_packed_size     = 0;
    attr->reg_cost             = ucs_linear_func_make(0, 0);
    memset(&attr->local_cpus, 0xff, sizeof(attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_sisci_mem_reg(uct_md_h md, void *address, size_t length,
                                     unsigned flags, uct_mem_h *memh_p)
{
    /* We have to emulate memory registration. Return dummy pointer */
    *memh_p = (void *) 0xdeadbeef;
    return UCS_OK;
}

static ucs_status_t uct_sisci_mem_dereg(uct_md_h uct_md,
                                       const uct_md_mem_dereg_params_t *params)
{
    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    ucs_assert(params->memh == (void*)0xdeadbeef);

    return UCS_OK;
}

static ucs_status_t uct_sisci_md_open(uct_component_t *component, const char *md_name,
                                     const uct_md_config_t *config, uct_md_h *md_p)
{

    uct_sisci_md_config_t *md_config = ucs_derived_of(config,
                                                     uct_sisci_md_config_t);

    static uct_md_ops_t md_ops = {
        .close              = ucs_empty_function,
        .query              = uct_sisci_md_query,
        .mkey_pack          = ucs_empty_function_return_success,
        .mem_reg            = uct_sisci_mem_reg,
        .mem_dereg          = uct_sisci_mem_dereg,
        .detect_memory_type = ucs_empty_function_return_unsupported
    };

    //create sisci memory domain struct
    //TODO, make it not full of poo poo
    static uct_sisci_md_t md;

    md.super.ops       = &md_ops;
    md.super.component = &uct_sisci_component;
    md.num_devices     = md_config->num_devices;

    *md_p = &md.super;

    //uct_md_h = sisci_md;

    //md_name = "sisci";



    printf("UCT_SISCI_MD_OPEN\n");
    return UCS_OK;
}


ucs_status_t uct_sisci_ep_put_short (uct_ep_h tl_ep, const void *buffer,
                                 unsigned length, uint64_t remote_addr,
                                 uct_rkey_t rkey)
{
    //TODO
    return UCS_ERR_NOT_IMPLEMENTED;
}

ssize_t uct_sisci_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                            void *arg, uint64_t remote_addr, uct_rkey_t rkey)
{
    //TODO
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_get_bcopy(uct_ep_h tl_ep, uct_unpack_callback_t unpack_cb,
                                 void *arg, size_t length,
                                 uint64_t remote_addr, uct_rkey_t rkey,
                                 uct_completion_t *comp)
{
    //TODO
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_atomic32_post(uct_ep_h ep, unsigned opcode, uint32_t value,
                                     uint64_t remote_addr, uct_rkey_t rkey)
{
    //TODO
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_atomic64_post(uct_ep_h ep, unsigned opcode, uint64_t value,
                                     uint64_t remote_addr, uct_rkey_t rkey)
{
    //TODO
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_atomic64_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                      uint64_t value, uint64_t *result,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp)
{
    //TODO
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_atomic32_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                      uint32_t value, uint32_t *result,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp)
{
    //TODO
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare,
                                      uint64_t swap, uint64_t remote_addr,
                                      uct_rkey_t rkey, uint64_t *result,
                                      uct_completion_t *comp)
{
    //TODO
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare,
                                      uint32_t swap, uint64_t remote_addr,
                                      uct_rkey_t rkey, uint32_t *result,
                                      uct_completion_t *comp)
{
    //TODO
    return UCS_ERR_NOT_IMPLEMENTED;
}

//from sm self.c

ucs_status_t uct_sisci_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                  const void *payload, unsigned length)
{
    //TODO
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_am_short_iov(uct_ep_h tl_ep, uint8_t id,
                                      const uct_iov_t *iov, size_t iovcnt)
{
    //TODO
    return UCS_ERR_NOT_IMPLEMENTED;
}

ssize_t uct_sisci_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                             uct_pack_callback_t pack_cb, void *arg,
                             unsigned flags)
{
    //TODO
    return 0;
}

static int uct_sisci_iface_is_reachable(const uct_iface_h tl_iface,
                                       const uct_device_addr_t *dev_addr,
                                       const uct_iface_addr_t *iface_addr)
{
    //TODO
    //const uct_self_iface_t     *iface = ucs_derived_of(tl_iface, uct_self_iface_t);
    //const uct_self_iface_addr_t *addr = (const uct_self_iface_addr_t*)iface_addr;

    //return (addr != NULL) && (iface->id == *addr);
    return 0;
}

static ucs_status_t uct_sisci_iface_get_address(uct_iface_h tl_iface,
                                               uct_iface_addr_t *addr)
{
    //TODO
    //const uct_self_iface_t *iface = ucs_derived_of(tl_iface, uct_self_iface_t);
    //*(uct_self_iface_addr_t*)addr = iface->id;
    return UCS_ERR_NOT_IMPLEMENTED;
}

static ucs_status_t uct_self_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *attr)
{
    return UCS_ERR_NOT_IMPLEMENTED;
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
        .prefix         = "SISCI_",
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
     

    .ep_put_short             = uct_sisci_ep_put_short,     // bap
    .ep_put_bcopy             = uct_sisci_ep_put_bcopy,     // bap
    .ep_get_bcopy             = uct_sisci_ep_get_bcopy,     // bap
    .ep_am_short              = uct_sisci_ep_am_short,      // bap
    .ep_am_short_iov          = uct_sisci_ep_am_short_iov,  // bap
    .ep_am_bcopy              = uct_sisci_ep_am_bcopy,      // bap
    .ep_atomic_cswap64        = uct_sisci_ep_atomic_cswap64,// bap
    .ep_atomic64_post         = uct_sisci_ep_atomic64_post, // bap
    .ep_atomic64_fetch        = uct_sisci_ep_atomic64_fetch,// bap
    .ep_atomic_cswap32        = uct_sisci_ep_atomic_cswap32,// bap
    .ep_atomic32_post         = uct_sisci_ep_atomic32_post, // bap
    .ep_atomic32_fetch        = uct_sisci_ep_atomic32_fetch,// bap
    .ep_flush                 = uct_base_ep_flush,          // maybe TODO, trenger vi å endre dette
    .ep_fence                 = uct_base_ep_fence,
    .ep_check                 = ucs_empty_function_return_success,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_sisci_ep_t),            //bapped
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_sisci_ep_t),         
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_sisci_iface_t),      //bapped
    .iface_query              = uct_self_iface_query,       //
    .iface_get_device_address = ucs_empty_function_return_success,
    .iface_get_address        = uct_sisci_iface_get_address, //
    .iface_is_reachable       = uct_sisci_iface_is_reachable //
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


