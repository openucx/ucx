
#include <ucs/type/class.h>
#include <ucs/type/status.h>
#include <ucs/sys/string.h>

#include "stdio.h"

#include "sisci.h"
#include "sisci_iface.h" //TODO, is this needed?
//#include "sisci_iface.c"


#define ADAPTER_NO 0
#define NO_FLAGS 0

/* Forward declarations */
static uct_iface_ops_t uct_sisci_iface_ops;
static uct_component_t uct_sisci_component;


static ucs_mpool_ops_t uct_sisci_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL,
    .obj_str       = NULL
};

static ucs_config_field_t uct_sisci_iface_config_table[] = {
    NULL
};


static ucs_config_field_t uct_sisci_md_config_table[] = {
    NULL
};

int sci_opened = 0;

/*
    The linux version initialization of the sisci api doesnt do much except for comparing the api version against the adapter version, and setting up some ref handles
    So we dont really need handle anything special except for the initialization faliing :). 
    Since the api doesn't have any good way to check if the driver is initialized we have to keep track of it ourselves. 
*/
static unsigned int uct_sci_open(){
    sci_error_t sci_error = 0;

    printf("sci_open(%d)\n", sci_opened);
    if (sci_opened == 0)
    {
        SCIInitialize(0,&sci_error);
        if (sci_error != SCI_ERR_OK)
        {
            printf("sci_init error: %s/n", SCIGetErrorString(sci_error));
            return 0;
        }
        sci_opened = 1;

    }
    return 1;
}


/*
    Closing the api is even more hands off than 
*/
static unsigned int uct_sci_close(){
    printf("sci_close(%d)\n", sci_opened);
    if (sci_opened == 1)
    {
        SCITerminate();
        sci_opened = 0;
    }
    return 1;    
}


void sisci_testing() {
    printf("Linking is correct to some degree :) \n");
}

//various "class" funcitons, don't really know how they work yet, but seems to be some sort of glue code. 
//also known as "macro hell"
static UCS_CLASS_CLEANUP_FUNC(uct_sisci_ep_t)
{
    printf("UCS_SICSCI_EP_CLEANUP_FUNC()\n");
}

static UCS_CLASS_INIT_FUNC(uct_sisci_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{

    unsigned int nodeID;
    unsigned int adapterID = 0;
    unsigned int flags = 0;
    size_t alignment;
    size_t align_offset;
    sci_error_t sci_error;
    ucs_status_t status;

    printf("UCS_SISCI_CLASS_INIT_FUNC() hm\n");

    UCS_CLASS_CALL_SUPER_INIT(
            uct_base_iface_t, &uct_sisci_iface_ops,
            &uct_base_iface_internal_ops, md, worker, params,
            tl_config UCS_STATS_ARG(
                    (params->field_mask & UCT_IFACE_PARAM_FIELD_STATS_ROOT) ?
                            params->stats_root :
                            NULL) UCS_STATS_ARG(UCT_SISCI_NAME));
    

    //---------- IFACE SISCI --------------------------
    SCIGetLocalNodeId(adapterID, &nodeID, flags, &sci_error);

    if (sci_error != SCI_ERR_OK) { 
        printf("SCI_IFACE_INIT: %s\n", SCIGetErrorString(sci_error));
    } 
    
    self->send_size = 1024;
    self->device_addr = nodeID;
    self->id = 13337;
    self->send_size = 65536; //this is probbably arbitrary, and could be higher. 2^16 was just selected for looks

    /*Need to find out how mpool works and how it is used by the underlying systems in ucx*/
    status = uct_iface_param_am_alignment(params, self->send_size, 0, 0,
                                          &alignment, &align_offset);
    if (status != UCS_OK) {
        printf("failed to init sisci mpool\n");
        return status;
    }

    status = ucs_mpool_init(
            &self->msg_mp, 0, self->send_size, align_offset, alignment,
            10, /* 2 elements are enough for most of communications */
            UINT_MAX, &uct_sisci_mpool_ops, "sisci_msg_desc");

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_sisci_iface_t)
{
    /* 
        TODO: Add proper cleanup for iface, i.e free resources that were allocated on init. 
    */
    printf("UCS_CLASS_CLEANUP_FUNC: SISCI_IFACE\n");
    ucs_mpool_cleanup(&self->msg_mp, 1);

}

UCS_CLASS_DEFINE(uct_sisci_iface_t, uct_base_iface_t);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_sisci_iface_t, uct_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_sisci_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);



static UCS_CLASS_INIT_FUNC(uct_sisci_ep_t, const uct_ep_params_t *params)
{
    //uct_sisci_iface_t *iface = ucs_derived_of(params->iface, uct_sisci_iface_t); //unused
    //uct_sisci_md_t *md = ucs_derived_of(iface->super.md, uct_sisci_md_t); //unused

    //make a segment;

    //sd : sci virtual device       md.sci_virtual_device
    //segment: local segment
    //segment id            
    //size                          md.segment_size
    //callback  
    //callbackarg   
    //flags                         0
    //error                         sci_error_t



    uct_sisci_iface_addr_t* iface_addr =  (uct_sisci_iface_addr_t*) params->iface_addr;
    uct_sisci_device_addr_t* dev_addr = (uct_sisci_device_addr_t*) params->dev_addr;

    unsigned int segment_id = (unsigned int) iface_addr->segment_id;
    unsigned int node_id = (unsigned int) dev_addr->node_id;
    

    printf("create_ep: nodeID: %d segID: %d\n", segment_id, node_id);
    self->super.super.iface = params->iface;
    self->remote_segment_id = segment_id;
    self->remote_node_id = node_id;

        

    return UCS_OK;
}

UCS_CLASS_DEFINE(uct_sisci_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_sisci_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_sisci_ep_t, uct_ep_t);




static ucs_status_t uct_sisci_query_md_resources(uct_component_t *component,
                                              uct_md_resource_desc_t **resources_p,
                                              unsigned int *num_resources_p)
{
    
    uct_md_resource_desc_t  *resources;
    int num_resources = 1;
    ucs_status_t status;

    resources = ucs_malloc(sizeof(*resources), "SCI resources");



    if(resources == NULL) {
        //TODO Handle memory errors.
        status = UCS_ERR_NO_MEMORY;
        printf("NO MEMORY\n");
    }

    *resources_p = resources;
    *num_resources_p = num_resources;

    status = UCS_OK;

    //ucs_snprintf_zero(resources->md_name, UCT_MD_NAME_MAX, "%s", component->name);

   
    printf("SISCI: UCT_SICI_QUERY_MD_RESOURCES\n");
    
    return status;
}


static ucs_status_t uct_sisci_query_devices(uct_md_h md,
                                   uct_tl_device_resource_t **devices_p,
                                   unsigned *num_devices_p)
{
    ucs_status_t status = -1;
    /*
        At this point its not clear if the memory domain has been opened yet.
        The memory domain is most likely opened.
    */


    /*
    Currently we are hard coding in the amount of devices and its properties.
    The reasoning for this is the rather "limited" scope of our master thesis,  
    */

    printf("UCT_SISCI_QUERY_DEVICES\n");

    /* 
        Taken from self.c, 
    */

    
    status = uct_single_device_resource(md, UCT_SISCI_NAME,
                                      UCT_DEVICE_TYPE_SHM,
                                      UCS_SYS_DEVICE_ID_UNKNOWN, devices_p,
                                      num_devices_p);
    
    printf("query_devices_status: %d\n", status);
    return status; 

    //return UCS_ERR_NO_DEVICE;
    //return UCS_OK;
}




static ucs_status_t uct_sisci_md_query(uct_md_h md, uct_md_attr_t *attr)
{
    /* Dummy memory registration provided. No real memory handling exists */
    //TODO: we have never looked into this 
    
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
    printf("uct_sisci_mem_reg()");

    /* We have to emulate memory registration. Return dummy pointer */
    *memh_p = (void *) 0xdeadbeef;
    return UCS_OK;
}

static ucs_status_t uct_sisci_mem_dereg(uct_md_h uct_md,
                                       const uct_md_mem_dereg_params_t *params)
{
    printf("uct_sisci_mem_dereg()");
    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    ucs_assert(params->memh == (void*)0xdeadbeef);

    return UCS_OK;
}


static void uct_sisci_md_close(uct_md_h md) {
    //TODO: Maybe free up all segments or something lmao
    
    uct_sisci_md_t * sci_md = ucs_derived_of(md, uct_sisci_md_t);
    sci_error_t sci_error;
    printf("uct_sisci_md_close: teehee %d\n", sci_md->segment_id);

    SCIClose(sci_md->sisci_virtual_device, 0 , &sci_error);

    if (sci_error != SCI_ERR_OK)
        {
            printf("sci_init error: %s/n", SCIGetErrorString(sci_error));
        }
    
    uct_sci_close();
}

static ucs_status_t uct_sisci_md_open(uct_component_t *component, const char *md_name,
                                     const uct_md_config_t *config, uct_md_h *md_p)
{
    /*This seems like the most reasonable place to call SCI_INIT, not sure when the memory domain is closed though : )*/
    uct_sisci_md_config_t *md_config = ucs_derived_of(config, uct_sisci_md_config_t);

    static uct_md_ops_t md_ops = {
        .close              = uct_sisci_md_close, //ucs_empty_function
        .query              = uct_sisci_md_query,
        .mkey_pack          = ucs_empty_function_return_success,
        .mem_reg            = uct_sisci_mem_reg,
        .mem_dereg          = uct_sisci_mem_dereg,
        .detect_memory_type = ucs_empty_function_return_unsupported
    };

    //create sisci memory domain struct
    //TODO, make it not full of poo poo
    static uct_sisci_md_t md;
    sci_error_t errors;

    uct_sci_open();

    SCIOpen(&md.sisci_virtual_device, 0, &errors);


    if (errors != SCI_ERR_OK)
        {
            printf("md_open error: %s/n", SCIGetErrorString(errors));
            return UCS_ERR_NO_RESOURCE;
        }
    
    //SCIClose(md.sisci_virtual_device, 0 , &errors);
    //uct_sci_close();

    md.super.ops       = &md_ops;
    md.super.component = &uct_sisci_component;
    md.num_devices     = md_config->num_devices;
    md.segment_id = 11;
    
    
    *md_p = &md.super;

    //uct_md_h = sisci_md;

    //md_name = "sisci";


    printf("number of devices : %ld \n", md.num_devices);
    printf("UCT_SISCI_MD_OPEN\n");
    return UCS_OK;
}

/*
static uct_sisci_md_t uct_sisci_md(){
    //empty struct
    //sci_error_t errors;
    //sci_desc_t sc_descriptor;

}*/

ucs_status_t uct_sisci_ep_put_short (uct_ep_h tl_ep, const void *buffer,
                                 unsigned length, uint64_t remote_addr,
                                 uct_rkey_t rkey)
{
    //TODO
    printf("uct_sisci_ep_put_short()\n");
    return UCS_ERR_NOT_IMPLEMENTED;
}

ssize_t uct_sisci_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                            void *arg, uint64_t remote_addr, uct_rkey_t rkey)
{
    //TODO
    printf("uct_sisci_ep_put_bcopy()\n");
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_get_bcopy(uct_ep_h tl_ep, uct_unpack_callback_t unpack_cb,
                                 void *arg, size_t length,
                                 uint64_t remote_addr, uct_rkey_t rkey,
                                 uct_completion_t *comp)
{
    //TODO
    printf("uct_sisci_ep_get_bcopy()\n");
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_atomic32_post(uct_ep_h ep, unsigned opcode, uint32_t value,
                                     uint64_t remote_addr, uct_rkey_t rkey)
{
    //TODO
    printf("uct_sisci_ep_atomic32_post()\n");
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_atomic64_post(uct_ep_h ep, unsigned opcode, uint64_t value,
                                     uint64_t remote_addr, uct_rkey_t rkey)
{
    //TODO
    printf("uct_sisci_ep_atomic64_post()\n");
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_atomic64_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                      uint64_t value, uint64_t *result,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp)
{
    //TODO
    printf("uct_sisci_ep_atomic64_fetch()\n");
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_atomic32_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                      uint32_t value, uint32_t *result,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp)
{
    //TODO
    printf("uct_sisci_ep_atomic32_fetch()\n");
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare,
                                      uint64_t swap, uint64_t remote_addr,
                                      uct_rkey_t rkey, uint64_t *result,
                                      uct_completion_t *comp)
{
    //TODO
    printf("uct_sisci_ep_atomic_cswap64()\n");
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare,
                                      uint32_t swap, uint64_t remote_addr,
                                      uct_rkey_t rkey, uint32_t *result,
                                      uct_completion_t *comp)
{
    //TODO
    printf("uct_sisci_ep_atomic_cswap32()\n");
    return UCS_ERR_NOT_IMPLEMENTED;
}

//from sm self.c

ucs_status_t uct_sisci_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                  const void *payload, unsigned length)
{
    //TODO
    printf("uct_sisci_ep_am_short() %d %ld %d \n", id, header, length);
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_sisci_ep_am_short_iov(uct_ep_h tl_ep, uint8_t id,
                                      const uct_iov_t *iov, size_t iovcnt)
{
    //TODO
    printf("uct_sisci_ep_am_short_iov()\n");
    return UCS_ERR_NOT_IMPLEMENTED;
}

ssize_t uct_sisci_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                             uct_pack_callback_t pack_cb, void *arg,
                             unsigned flags)
{
    //TODO
    printf("uct_sisci_ep_am_bcopy()\n");
    return -8;
}

ucs_status_t uct_sisci_ep_am_zcopy(uct_ep_h ep, uint8_t id, const void *header, unsigned header_length, 
                            const uct_iov_t *iov, size_t iovcnt, unsigned flags, uct_completion_t *comp) 
{
    printf("uct_sisci_ep_am_zcopy()\n");
    return UCS_ERR_NOT_IMPLEMENTED;    
}

int uct_sisci_iface_is_reachable(const uct_iface_h tl_iface,
                                       const uct_device_addr_t *dev_addr,
                                       const uct_iface_addr_t *iface_addr)
{
    //TODO make not die
    
    //const uct_self_iface_t     *iface = ucs_derived_of(tl_iface, uct_self_iface_t);
    //const uct_self_iface_addr_t *addr = (const uct_self_iface_addr_t*)iface_addr;

    //return (addr != NULL) && (iface->id == *addr);
    printf("UCT_sisci_iface_is reachable\n");
    return 1;
}

ucs_status_t uct_sisci_get_device_address(uct_iface_h iface, uct_device_addr_t *addr) {
    
    uct_sisci_iface_t* sisci_iface = ucs_derived_of(iface, uct_sisci_iface_t);
    
    uct_sisci_md_t* md =  ucs_derived_of(sisci_iface->super.md, uct_sisci_md_t);  

    uct_sisci_device_addr_t* sci_addr = (uct_sisci_device_addr_t *) addr;

    printf("iface_data = %d %d\n", sisci_iface->id, sisci_iface->device_addr);
    printf("sisci_get_device_address() %d\n", md->segment_id);

    sci_addr->node_id = sisci_iface->device_addr;

    return UCS_OK;
}

ucs_status_t uct_sisci_iface_get_address(uct_iface_h tl_iface,
                                               uct_iface_addr_t *addr)
{
    //TODO: Don't lie, but get iface_addr from config.
    
    uct_sisci_iface_t* iface = ucs_derived_of(tl_iface, uct_sisci_iface_t);

    uct_sisci_iface_addr_t* iface_addr = (uct_sisci_iface_addr_t *) addr;
    
    printf("uct_iface_get_address()\n");
    iface_addr->segment_id = iface->id;
    
    return UCS_OK;
}

static ucs_status_t uct_sisci_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *attr)
{
    
    printf("UCT_sisci_iface_query\n");

    //TODO: insert necessarry lies to make ucx want us.
    //taken from uct_iface.c sets default attributes to zero.
    memset(attr, 0, sizeof(*attr));


    /*  Start of lies  */
    attr->dev_num_paths = 1;
    attr->max_num_eps = 32;    
    
<<<<<<< HEAD
    attr->cap.flags =           UCT_IFACE_FLAG_CONNECT_TO_IFACE | 
                                UCT_IFACE_FLAG_CONNECT_TO_EP    |
                                UCT_IFACE_FLAG_AM_BCOPY         | 
                                UCT_IFACE_FLAG_AM_SHORT         | 
                                UCT_IFACE_FLAG_AM_ZCOPY;
    attr->cap.event_flags  =    UCT_IFACE_FLAG_EVENT_SEND_COMP |
                                UCT_IFACE_FLAG_EVENT_RECV;
=======
    attr->cap.flags =   UCT_IFACE_FLAG_CONNECT_TO_IFACE | 
                        UCT_IFACE_FLAG_CONNECT_TO_EP    |
                        UCT_IFACE_FLAG_AM_SHORT         |
                        UCT_IFACE_FLAG_CB_SYNC          |
                        UCT_IFACE_FLAG_AM_BCOPY         | 
                        UCT_IFACE_FLAG_AM_ZCOPY;
    attr->cap.event_flags  = UCT_IFACE_FLAG_EVENT_SEND_COMP |
<<<<<<< HEAD
                             UCT_IFACE_FLAG_EVENT_RECV;
>>>>>>> 42697b26573c87265e773f902ccf8d6dbb966b13
=======
                             UCT_IFACE_FLAG_EVENT_RECV      |
<<<<<<< HEAD
                             UCT_IFACE_FLAG_EVENT_ASYNC_CB;
>>>>>>> 15e028b5baca0f5bdc21a443ea7755f9e376cb29
=======
                             UCT_IFACE_FLAG_EVENT_ASYNC_CB  |
                             UCT_IFACE_FLAG_EVENT_RECV_SIG;
>>>>>>> f2b001f1f685e361fcf53744da91289262da29ec

    attr->device_addr_len  = sizeof(uct_sisci_device_addr_t);
    attr->ep_addr_len      = sizeof(uct_sicsci_ep_addr_t);
    attr->iface_addr_len   = sizeof(uct_sisci_iface_addr_t);
    
    
    //TODO: sane numbers, no lies.
    /* AM flags - TODO: these might need to be fine tuned at a later stage */
    attr->cap.am.max_short = 128;
    attr->cap.am.max_bcopy = 128;
    attr->cap.am.min_zcopy = 256;
    attr->cap.am.max_zcopy = 1024;


    attr->latency                 = ucs_linear_func_make(0, 0);
    attr->bandwidth.dedicated     = 6911.0 * UCS_MBYTE;
    attr->bandwidth.shared        = 0;
    attr->overhead                = 10e-9;
    attr->priority                = 0;



    printf("iface->attr->cap.flags: %ld event_flags-> %ld\n", attr->cap.flags, attr->cap.event_flags);
    return UCS_OK;
    //return UCS_ERR_NOT_IMPLEMENTED;
}


static ucs_status_t uct_sisci_md_rkey_unpack(uct_component_t *component,
                                            const void *rkey_buffer, uct_rkey_t *rkey_p,
                                            void **handle_p)
{
    /**
     * Pseudo stub function for the key unpacking
     * Need rkey == 0 due to work with same process to reuse uct_base_[put|get|atomic]*
     */
    printf("uct_sisci_md_rkey_unpack()");
    *rkey_p   = 0;
    *handle_p = NULL;
    return UCS_OK;
}

/*
    TODO: Figure out what to change the commented lines to : )
*/
static uct_component_t uct_sisci_component = {
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
    .ep_am_zcopy              = uct_sisci_ep_am_zcopy,
    .ep_atomic_cswap64        = uct_sisci_ep_atomic_cswap64,// bap
    .ep_atomic64_post         = uct_sisci_ep_atomic64_post, // bap
    .ep_atomic64_fetch        = uct_sisci_ep_atomic64_fetch,// bap
    .ep_atomic_cswap32        = uct_sisci_ep_atomic_cswap32,// bap
    .ep_atomic32_post         = uct_sisci_ep_atomic32_post, // bap
    .ep_atomic32_fetch        = uct_sisci_ep_atomic32_fetch,// bap
    .ep_flush                 = uct_base_ep_flush,          // maybe TODO, trenger vi å endre dette
    .ep_fence                 = uct_base_ep_fence,          // covered av uct base
    .ep_check                 = ucs_empty_function_return_success,  //covered tror jeg
    .ep_pending_add           = ucs_empty_function_return_busy,     //covered
    .ep_pending_purge         = ucs_empty_function,                 //covered
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_sisci_ep_t),            //bapped? is makro hell
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_sisci_ep_t),         //more makro hell
    .iface_flush              = uct_base_iface_flush,           //covered av uct base
    .iface_fence              = uct_base_iface_fence,           //covered av uct base
    .iface_progress_enable    = ucs_empty_function,             //covered
    .iface_progress_disable   = ucs_empty_function,             //covered
    .iface_event_arm          = ucs_empty_function_return_success,
    .iface_progress           = ucs_empty_function_return_zero, //covered
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_sisci_iface_t),      //bapped more makro hell
    .iface_query              = uct_sisci_iface_query,       //bap
    .iface_get_device_address = uct_sisci_get_device_address, //covered
    .iface_get_address        = uct_sisci_iface_get_address, // bap
    .iface_is_reachable       = uct_sisci_iface_is_reachable // bap
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


