/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2015-2019.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "hello_world_util.h"
#include <limits.h>

#include <uct/api/uct.h>

#include <assert.h>
#include <inttypes.h>


typedef enum {
    FUNC_AM_SHORT,
    FUNC_AM_BCOPY,
    FUNC_AM_ZCOPY
} func_am_t;

typedef struct {
    int  is_uct_desc;
} recv_desc_t;

typedef struct {
    char               *server_name;
    uint16_t            server_port;
    sa_family_t         ai_family;
    func_am_t           func_am_type;
    const char         *dev_name;
    const char         *tl_name;
    long                test_strlen;
} cmd_args_t;

typedef struct {
    uct_iface_attr_t    iface_attr; /* Interface attributes: capabilities and limitations */
    uct_iface_h         iface;      /* Communication interface context */
    uct_md_attr_t       md_attr;    /* Memory domain attributes: capabilities and limitations */
    uct_md_h            md;         /* Memory domain */
    uct_worker_h        worker;     /* Workers represent allocated resources in a communication thread */
} iface_info_t;

/* Helper data type for am_short */
typedef struct {
    uint64_t            header;
    char               *payload;
    size_t              len;
} am_short_args_t;

/* Helper data type for am_bcopy */
typedef struct {
    char               *data;
    size_t              len;
} am_bcopy_args_t;

/* Helper data type for am_zcopy */
typedef struct {
    uct_completion_t    uct_comp;
    uct_md_h            md;
    uct_mem_h           memh;
} zcopy_comp_t;

static void* desc_holder = NULL;

int print_err_usage(void);

static char *func_am_t_str(func_am_t func_am_type)
{
    switch (func_am_type) {
    case FUNC_AM_SHORT:
        return "uct_ep_am_short";
    case FUNC_AM_BCOPY:
        return "uct_ep_am_bcopy";
    case FUNC_AM_ZCOPY:
        return "uct_ep_am_zcopy";
    }
    return NULL;
}

static size_t func_am_max_size(func_am_t func_am_type,
                               const uct_iface_attr_t *attr)
{
    switch (func_am_type) {
    case FUNC_AM_SHORT:
        return attr->cap.am.max_short;
    case FUNC_AM_BCOPY:
        return attr->cap.am.max_bcopy;
    case FUNC_AM_ZCOPY:
        return attr->cap.am.max_zcopy;
    }
    return 0;
}

/* Helper function for am_short */
void am_short_params_pack(char *buf, size_t len, am_short_args_t *args)
{
    args->header      = *(uint64_t *)buf;
    if (len > sizeof(args->header)) {
        args->payload = (buf + sizeof(args->header));
        args->len     = len - sizeof(args->header);
    } else {
        args->payload = NULL;
        args->len     = 0;
    }
}

ucs_status_t do_am_short(iface_info_t *if_info, uct_ep_h ep, uint8_t id,
                         const cmd_args_t *cmd_args, char *buf)
{
    ucs_status_t    status;
    am_short_args_t send_args;

    am_short_params_pack(buf, cmd_args->test_strlen, &send_args);

    do {
        /* Send active message to remote endpoint */
        status = uct_ep_am_short(ep, id, send_args.header, send_args.payload,
                                 send_args.len);
        uct_worker_progress(if_info->worker);
    } while (status == UCS_ERR_NO_RESOURCE);

    return status;
}

/* Pack callback for am_bcopy */
size_t am_bcopy_data_pack_cb(void *dest, void *arg)
{
    am_bcopy_args_t *bc_args = arg;
    mem_type_memcpy(dest, bc_args->data, bc_args->len);
    return bc_args->len;
}

ucs_status_t do_am_bcopy(iface_info_t *if_info, uct_ep_h ep, uint8_t id,
                         const cmd_args_t *cmd_args, char *buf)
{
    am_bcopy_args_t args;
    ssize_t len;

    args.data = buf;
    args.len  = cmd_args->test_strlen;

    /* Send active message to remote endpoint */
    do {
        len = uct_ep_am_bcopy(ep, id, am_bcopy_data_pack_cb, &args, 0);
        uct_worker_progress(if_info->worker);
    } while (len == UCS_ERR_NO_RESOURCE);
    /* Negative len is an error code */
    return (len >= 0) ? UCS_OK : (ucs_status_t)len;
}

/* Completion callback for am_zcopy */
void zcopy_completion_cb(uct_completion_t *self)
{
    zcopy_comp_t *comp = (zcopy_comp_t *)self;
    assert((comp->uct_comp.count == 0) && (self->status == UCS_OK));
    if (comp->memh != UCT_MEM_HANDLE_NULL) {
        uct_md_mem_dereg(comp->md, comp->memh);
    }
    desc_holder = (void *)0xDEADBEEF;
}

ucs_status_t do_am_zcopy(iface_info_t *if_info, uct_ep_h ep, uint8_t id,
                         const cmd_args_t *cmd_args, char *buf)
{
    ucs_status_t status = UCS_OK;
    uct_mem_h memh;
    uct_iov_t iov;
    zcopy_comp_t comp;

    if (if_info->md_attr.cap.flags & UCT_MD_FLAG_NEED_MEMH) {
        status = uct_md_mem_reg(if_info->md, buf, cmd_args->test_strlen,
                                UCT_MD_MEM_ACCESS_RMA, &memh);
    } else {
        memh = UCT_MEM_HANDLE_NULL;
    }

    iov.buffer = buf;
    iov.length = cmd_args->test_strlen;
    iov.memh   = memh;
    iov.stride = 0;
    iov.count  = 1;

    comp.uct_comp.func   = zcopy_completion_cb;
    comp.uct_comp.count  = 1;
    comp.uct_comp.status = UCS_OK;
    comp.md              = if_info->md;
    comp.memh            = memh;

    if (status == UCS_OK) {
        do {
            status = uct_ep_am_zcopy(ep, id, NULL, 0, &iov, 1, 0,
                                     (uct_completion_t *)&comp);
            uct_worker_progress(if_info->worker);
        } while (status == UCS_ERR_NO_RESOURCE);

        if (status == UCS_INPROGRESS) {
            while (!desc_holder) {
                /* Explicitly progress outstanding active message request */
                uct_worker_progress(if_info->worker);
            }
            status = UCS_OK;
        }
    }
    return status;
}
static void print_strings(const char *label, const char *local_str,
                          const char *remote_str, size_t length)
{
    fprintf(stdout, "\n\n----- UCT TEST SUCCESS ----\n\n");
    fprintf(stdout, "[%s] %s sent %s (%" PRIu64 " bytes)", label, local_str,
            (length != 0) ? remote_str : "<none>", length);
    fprintf(stdout, "\n\n---------------------------\n");
    fflush(stdout);
}

/* Callback to handle receive active message */
static ucs_status_t hello_world(void *arg, void *data, size_t length,
                                unsigned flags)
{
    func_am_t func_am_type = *(func_am_t *)arg;
    recv_desc_t *rdesc;

    print_strings("callback", func_am_t_str(func_am_type), data, length);

    if (flags & UCT_CB_PARAM_FLAG_DESC) {
        rdesc = (recv_desc_t *)data - 1;
        /* Hold descriptor to release later and return UCS_INPROGRESS */
        rdesc->is_uct_desc = 1;
        desc_holder = rdesc;
        return UCS_INPROGRESS;
    }

    /* We need to copy-out data and return UCS_OK if want to use the data
     * outside the callback */
    rdesc = malloc(sizeof(*rdesc) + length);
    CHKERR_ACTION(rdesc == NULL, "allocate memory\n", return UCS_ERR_NO_MEMORY);
    rdesc->is_uct_desc = 0;
    memcpy(rdesc + 1, data, length);
    desc_holder = rdesc;
    return UCS_OK;
}

/* Init the transport by its name */
static ucs_status_t init_iface(char *dev_name, char *tl_name,
                               func_am_t func_am_type,
                               iface_info_t *iface_p)
{
    ucs_status_t        status;
    uct_iface_config_t  *config; /* Defines interface configuration options */
    uct_iface_params_t  params;

    params.field_mask           = UCT_IFACE_PARAM_FIELD_OPEN_MODE   |
                                  UCT_IFACE_PARAM_FIELD_DEVICE      |
                                  UCT_IFACE_PARAM_FIELD_STATS_ROOT  |
                                  UCT_IFACE_PARAM_FIELD_RX_HEADROOM |
                                  UCT_IFACE_PARAM_FIELD_CPU_MASK;
    params.open_mode            = UCT_IFACE_OPEN_MODE_DEVICE;
    params.mode.device.tl_name  = tl_name;
    params.mode.device.dev_name = dev_name;
    params.stats_root           = NULL;
    params.rx_headroom          = sizeof(recv_desc_t);

    UCS_CPU_ZERO(&params.cpu_mask);
    /* Read transport-specific interface configuration */
    status = uct_md_iface_config_read(iface_p->md, tl_name, NULL, NULL, &config);
    CHKERR_JUMP(UCS_OK != status, "setup iface_config", error_ret);

    /* Open communication interface */
    assert(iface_p->iface == NULL);
    status = uct_iface_open(iface_p->md, iface_p->worker, &params, config,
                            &iface_p->iface);
    uct_config_release(config);
    CHKERR_JUMP(UCS_OK != status, "open temporary interface", error_ret);

    /* Enable progress on the interface */
    uct_iface_progress_enable(iface_p->iface,
                              UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    /* Get interface attributes */
    status = uct_iface_query(iface_p->iface, &iface_p->iface_attr);
    CHKERR_JUMP(UCS_OK != status, "query iface", error_iface);

    /* Check if current device and transport support required active messages */
    if ((func_am_type == FUNC_AM_SHORT) &&
        (iface_p->iface_attr.cap.flags & UCT_IFACE_FLAG_AM_SHORT)) {
        if (test_mem_type != UCS_MEMORY_TYPE_CUDA) {
            return UCS_OK;
        } else {
            fprintf(stderr, "AM short protocol doesn't support CUDA memory");
        }
    }

    if ((func_am_type == FUNC_AM_BCOPY) &&
        (iface_p->iface_attr.cap.flags & UCT_IFACE_FLAG_AM_BCOPY)) {
        return UCS_OK;
    }

    if ((func_am_type == FUNC_AM_ZCOPY) &&
        (iface_p->iface_attr.cap.flags & UCT_IFACE_FLAG_AM_ZCOPY)) {
        return UCS_OK;
    }

error_iface:
    uct_iface_close(iface_p->iface);
    iface_p->iface = NULL;
error_ret:
    return UCS_ERR_UNSUPPORTED;
}

/* Device and transport to be used are determined by minimum latency */
static ucs_status_t dev_tl_lookup(const cmd_args_t *cmd_args,
                                  iface_info_t *iface_p)
{
    uct_tl_resource_desc_t *tl_resources    = NULL; /* Communication resource descriptor */
    unsigned               num_tl_resources = 0;    /* Number of transport resources resource objects created */
    uct_component_h        *components;
    unsigned               num_components;
    unsigned               cmpt_index;
    uct_component_attr_t   component_attr;
    unsigned               md_index;
    unsigned               tl_index;
    uct_md_config_t        *md_config;
    ucs_status_t           status;

    status = uct_query_components(&components, &num_components);
    CHKERR_JUMP(UCS_OK != status, "query for components", error_ret);

    for (cmpt_index = 0; cmpt_index < num_components; ++cmpt_index) {

        component_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT;
        status = uct_component_query(components[cmpt_index], &component_attr);
        CHKERR_JUMP(UCS_OK != status, "query component attributes",
                    release_component_list);

        component_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
        component_attr.md_resources = alloca(sizeof(*component_attr.md_resources) *
                                             component_attr.md_resource_count);
        status = uct_component_query(components[cmpt_index], &component_attr);
        CHKERR_JUMP(UCS_OK != status, "query for memory domain resources",
                    release_component_list);

        iface_p->iface = NULL;

        /* Iterate through memory domain resources */
        for (md_index = 0; md_index < component_attr.md_resource_count; ++md_index) {
            status = uct_md_config_read(components[cmpt_index], NULL, NULL,
                                        &md_config);
            CHKERR_JUMP(UCS_OK != status, "read MD config",
                        release_component_list);

            status = uct_md_open(components[cmpt_index],
                                 component_attr.md_resources[md_index].md_name,
                                 md_config, &iface_p->md);
            uct_config_release(md_config);
            CHKERR_JUMP(UCS_OK != status, "open memory domains",
                        release_component_list);

            status = uct_md_query(iface_p->md, &iface_p->md_attr);
            CHKERR_JUMP(UCS_OK != status, "query iface",
                        close_md);

            status = uct_md_query_tl_resources(iface_p->md, &tl_resources,
                                               &num_tl_resources);
            CHKERR_JUMP(UCS_OK != status, "query transport resources", close_md);

            /* Go through each available transport and find the proper name */
            for (tl_index = 0; tl_index < num_tl_resources; ++tl_index) {
                if (!strcmp(cmd_args->dev_name, tl_resources[tl_index].dev_name) &&
                    !strcmp(cmd_args->tl_name, tl_resources[tl_index].tl_name)) {
                    if ((cmd_args->func_am_type == FUNC_AM_ZCOPY) &&
                        !(iface_p->md_attr.cap.reg_mem_types &
                          UCS_BIT(test_mem_type))) {
                        fprintf(stderr, "Unsupported memory type %s by "
                                UCT_TL_RESOURCE_DESC_FMT" on %s MD\n",
                                ucs_memory_type_names[test_mem_type],
                                UCT_TL_RESOURCE_DESC_ARG(&tl_resources[tl_index]),
                                component_attr.md_resources[md_index].md_name);
                        status = UCS_ERR_UNSUPPORTED;
                        break;
                    }

                    status = init_iface(tl_resources[tl_index].dev_name,
                                        tl_resources[tl_index].tl_name,
                                        cmd_args->func_am_type, iface_p);
                    if (status != UCS_OK) {
                        break;
                    }

                    fprintf(stdout, "Using "UCT_TL_RESOURCE_DESC_FMT"\n",
                            UCT_TL_RESOURCE_DESC_ARG(&tl_resources[tl_index]));
                    goto release_tl_resources;
                }
            }

release_tl_resources:
            uct_release_tl_resource_list(tl_resources);
            if ((status == UCS_OK) &&
                (tl_index < num_tl_resources)) {
                goto release_component_list;
            }

            tl_resources     = NULL;
            num_tl_resources = 0;
            uct_md_close(iface_p->md);
        }
    }

    fprintf(stderr, "No supported (dev/tl) found (%s/%s)\n",
            cmd_args->dev_name, cmd_args->tl_name);
    status = UCS_ERR_UNSUPPORTED;

release_component_list:
    uct_release_component_list(components);
error_ret:
    return status;
close_md:
    uct_md_close(iface_p->md);
    goto release_component_list;
}

int print_err_usage()
{
    const char func_template[] = "  -%c      Select \"%s\" function to send the message%s\n";

    fprintf(stderr, "Usage: uct_hello_world [parameters]\n");
    fprintf(stderr, "UCT hello world client/server example utility\n");
    fprintf(stderr, "\nParameters are:\n");
    fprintf(stderr, func_template, 'i', func_am_t_str(FUNC_AM_SHORT), " (default)");
    fprintf(stderr, func_template, 'b', func_am_t_str(FUNC_AM_BCOPY), "");
    fprintf(stderr, func_template, 'z', func_am_t_str(FUNC_AM_ZCOPY), "");
    fprintf(stderr, "  -d        Select device name\n");
    fprintf(stderr, "  -t        Select transport layer\n");
    fprintf(stderr, "  -n <name> Set node name or IP address "
            "of the server (required for client and should be ignored "
            "for server)\n");
    print_common_help();
    fprintf(stderr, "\nExample:\n");
    fprintf(stderr, "  Server: uct_hello_world -d eth0 -t tcp\n");
    fprintf(stderr, "  Client: uct_hello_world -d eth0 -t tcp -n localhost\n");

    return UCS_ERR_UNSUPPORTED;
}

int parse_cmd(int argc, char * const argv[], cmd_args_t *args)
{
    int c = 0, idx = 0;

    assert(args);
    memset(args, 0, sizeof(*args));

    /* Defaults */
    args->server_port   = 13337;
    args->ai_family     = AF_INET;
    args->func_am_type  = FUNC_AM_SHORT;
    args->test_strlen   = 16;

    while ((c = getopt(argc, argv, "6ibzd:t:n:p:s:m:h")) != -1) {
        switch (c) {
        case 'i':
            args->func_am_type = FUNC_AM_SHORT;
            break;
        case 'b':
            args->func_am_type = FUNC_AM_BCOPY;
            break;
        case 'z':
            args->func_am_type = FUNC_AM_ZCOPY;
            break;
        case 'd':
            args->dev_name = optarg;
            break;
        case 't':
            args->tl_name = optarg;
            break;
        case 'n':
            args->server_name = optarg;
            break;
        case '6':
            args->ai_family = AF_INET6;
            break;
        case 'p':
            args->server_port = atoi(optarg);
            if (args->server_port <= 0) {
                fprintf(stderr, "Wrong server port number %d\n",
                        args->server_port);
                return UCS_ERR_UNSUPPORTED;
            }
            break;
        case 's':
            args->test_strlen = atol(optarg);
            if (args->test_strlen < 0) {
                fprintf(stderr, "Wrong string size %ld\n", args->test_strlen);
                return UCS_ERR_UNSUPPORTED;
            }
            break;
        case 'm':
            test_mem_type = parse_mem_type(optarg);
            if (test_mem_type == UCS_MEMORY_TYPE_LAST) {
                return UCS_ERR_UNSUPPORTED;
            }
            break;
        case 'h':
        default:
            return print_err_usage();
        }
    }
    fprintf(stderr, "INFO: UCT_HELLO_WORLD AM function = %s server = %s port = %d\n",
            func_am_t_str(args->func_am_type), args->server_name,
            args->server_port);

    for (idx = optind; idx < argc; idx++) {
        fprintf(stderr, "WARNING: Non-option argument %s\n", argv[idx]);
    }

    if (args->dev_name == NULL) {
        fprintf(stderr, "WARNING: device is not set\n");
        return print_err_usage();
    }

    if (args->tl_name == NULL) {
        fprintf(stderr, "WARNING: transport layer is not set\n");
        return print_err_usage();
    }

    return UCS_OK;
}

/* The caller is responsible to free *rbuf */
int sendrecv(int sock, const void *sbuf, size_t slen, void **rbuf)
{
    int ret = 0;
    size_t rlen = 0;
    *rbuf = NULL;

    ret = send(sock, &slen, sizeof(slen), 0);
    if ((ret < 0) || (ret != sizeof(slen))) {
        fprintf(stderr, "failed to send buffer length\n");
        return -1;
    }

    ret = send(sock, sbuf, slen, 0);
    if (ret != (int)slen) {
        fprintf(stderr, "failed to send buffer, return value %d\n", ret);
        return -1;
    }

    ret = recv(sock, &rlen, sizeof(rlen), MSG_WAITALL);
    if ((ret != sizeof(rlen)) || (rlen > (SIZE_MAX / 2))) {
        fprintf(stderr,
                "failed to receive device address length, return value %d\n",
                ret);
        return -1;
    }

    *rbuf = calloc(1, rlen);
    if (!*rbuf) {
        fprintf(stderr, "failed to allocate receive buffer\n");
        return -1;
    }

    ret = recv(sock, *rbuf, rlen, MSG_WAITALL);
    if (ret != (int)rlen) {
        fprintf(stderr, "failed to receive device address, return value %d\n",
                ret);
        return -1;
    }

    return 0;
}

static void progress_worker(void *arg)
{
    uct_worker_progress((uct_worker_h)arg);
}

int main(int argc, char **argv)
{
    uct_device_addr_t   *peer_dev   = NULL;
    uct_iface_addr_t    *peer_iface = NULL;
    uct_ep_addr_t       *own_ep     = NULL;
    uct_ep_addr_t       *peer_ep    = NULL;
    uint8_t             id          = 0;
    int                 oob_sock    = -1;  /* OOB connection socket */
    ucs_status_t        status      = UCS_OK; /* status codes for UCS */
    uct_device_addr_t   *own_dev;
    uct_iface_addr_t    *own_iface;
    uct_ep_h            ep;                   /* Remote endpoint */
    ucs_async_context_t *async;               /* Async event context manages
                                                 times and fd notifications */
    cmd_args_t          cmd_args;
    iface_info_t        if_info;
    uct_ep_params_t     ep_params;
    int                 res;

    /* Parse the command line */
    if (parse_cmd(argc, argv, &cmd_args)) {
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    /* Initialize context
     * It is better to use different contexts for different workers */
    status = ucs_async_context_create(UCS_ASYNC_MODE_THREAD_SPINLOCK, &async);
    CHKERR_JUMP(UCS_OK != status, "init async context", out);

    /* Create a worker object */
    status = uct_worker_create(async, UCS_THREAD_MODE_SINGLE, &if_info.worker);
    CHKERR_JUMP(UCS_OK != status, "create worker", out_cleanup_async);

    /* Search for the desired transport */
    status = dev_tl_lookup(&cmd_args, &if_info);
    CHKERR_JUMP(UCS_OK != status, "find supported device and transport",
                out_destroy_worker);

    /* Set active message handler */
    status = uct_iface_set_am_handler(if_info.iface, id, hello_world,
                                      &cmd_args.func_am_type, 0);
    CHKERR_JUMP(UCS_OK != status, "set callback", out_destroy_iface);

    own_dev = (uct_device_addr_t*)calloc(1, if_info.iface_attr.device_addr_len);
    CHKERR_JUMP(NULL == own_dev, "allocate memory for dev addr",
                out_destroy_iface);

    own_iface = (uct_iface_addr_t*)calloc(1, if_info.iface_attr.iface_addr_len);
    CHKERR_JUMP(NULL == own_iface, "allocate memory for if addr",
                out_free_dev_addrs);

    oob_sock = connect_common(cmd_args.server_name, cmd_args.server_port,
                              cmd_args.ai_family);

    CHKERR_ACTION(oob_sock < 0, "OOB connect",
                  status = UCS_ERR_IO_ERROR; goto out_close_oob_sock);

    /* Get device address */
    if (if_info.iface_attr.device_addr_len > 0) {
        status = uct_iface_get_device_address(if_info.iface, own_dev);
        CHKERR_JUMP(UCS_OK != status, "get device address", out_free_if_addrs);

        res = sendrecv(oob_sock, own_dev, if_info.iface_attr.device_addr_len,
                       (void**)&peer_dev);
        CHKERR_ACTION(0 != res, "device exchange", status = UCS_ERR_NO_MESSAGE;
                      goto out_close_oob_sock);
    }

    /* Get interface address */
    if (if_info.iface_attr.iface_addr_len > 0) {
        status = uct_iface_get_address(if_info.iface, own_iface);
        CHKERR_JUMP(UCS_OK != status, "get interface address",
                    out_close_oob_sock);

        status = (ucs_status_t)sendrecv(oob_sock, own_iface, if_info.iface_attr.iface_addr_len,
                                        (void **)&peer_iface);
        CHKERR_JUMP(0 != status, "ifaces exchange", out_close_oob_sock);
    }

    status = (ucs_status_t)uct_iface_is_reachable(if_info.iface, peer_dev,
                                                  peer_iface);
    CHKERR_JUMP(0 == status, "reach the peer", out_close_oob_sock);

    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    ep_params.iface      = if_info.iface;
    if (if_info.iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        own_ep = (uct_ep_addr_t*)calloc(1, if_info.iface_attr.ep_addr_len);
        CHKERR_ACTION(NULL == own_ep, "allocate memory for ep addrs",
                      status = UCS_ERR_NO_MEMORY; goto out_close_oob_sock);

        /* Create new endpoint */
        status = uct_ep_create(&ep_params, &ep);
        CHKERR_JUMP(UCS_OK != status, "create endpoint", out_free_ep_addrs);

        /* Get endpoint address */
        status = uct_ep_get_address(ep, own_ep);
        CHKERR_JUMP(UCS_OK != status, "get endpoint address", out_free_ep);

        status = (ucs_status_t)sendrecv(oob_sock, own_ep, if_info.iface_attr.ep_addr_len,
                                        (void **)&peer_ep);
        CHKERR_JUMP(0 != status, "EPs exchange", out_free_ep);

        /* Connect endpoint to a remote endpoint */
        status = uct_ep_connect_to_ep(ep, peer_dev, peer_ep);
        if (barrier(oob_sock, progress_worker, if_info.worker)) {
            status = UCS_ERR_IO_ERROR;
            goto out_free_ep;
        }
    } else if (if_info.iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        /* Create an endpoint which is connected to a remote interface */
        ep_params.field_mask |= UCT_EP_PARAM_FIELD_DEV_ADDR |
                                UCT_EP_PARAM_FIELD_IFACE_ADDR;
        ep_params.dev_addr    = peer_dev;
        ep_params.iface_addr  = peer_iface;
        status = uct_ep_create(&ep_params, &ep);
        CHKERR_JUMP(UCS_OK != status, "create endpoint", out_free_ep_addrs);
    } else {
        status = UCS_ERR_UNSUPPORTED;
        goto out_free_ep_addrs;
    }

    if (cmd_args.test_strlen > func_am_max_size(cmd_args.func_am_type, &if_info.iface_attr)) {
        status = UCS_ERR_UNSUPPORTED;
        fprintf(stderr, "Test string is too long: %ld, max supported: %lu\n",
                cmd_args.test_strlen,
                func_am_max_size(cmd_args.func_am_type, &if_info.iface_attr));
        goto out_free_ep;
    }

    if (cmd_args.server_name) {
        char *str = (char *)mem_type_malloc(cmd_args.test_strlen);
        CHKERR_ACTION(str == NULL, "allocate memory",
                      status = UCS_ERR_NO_MEMORY; goto out_free_ep);
        res = generate_test_string(str, cmd_args.test_strlen);
        CHKERR_ACTION(res < 0, "generate test string",
                      status = UCS_ERR_NO_MEMORY; goto out_free_ep);

        /* Send active message to remote endpoint */
        if (cmd_args.func_am_type == FUNC_AM_SHORT) {
            status = do_am_short(&if_info, ep, id, &cmd_args, str);
        } else if (cmd_args.func_am_type == FUNC_AM_BCOPY) {
            status = do_am_bcopy(&if_info, ep, id, &cmd_args, str);
        } else if (cmd_args.func_am_type == FUNC_AM_ZCOPY) {
            status = do_am_zcopy(&if_info, ep, id, &cmd_args, str);
        }

        mem_type_free(str);
        CHKERR_JUMP(UCS_OK != status, "send active msg", out_free_ep);
    } else {
        recv_desc_t *rdesc;

        while (desc_holder == NULL) {
            /* Explicitly progress any outstanding active message requests */
            uct_worker_progress(if_info.worker);
        }

        rdesc = desc_holder;
        print_strings("main", func_am_t_str(cmd_args.func_am_type),
                      (char *)(rdesc + 1), cmd_args.test_strlen);

        if (rdesc->is_uct_desc) {
            /* Release descriptor because callback returns UCS_INPROGRESS */
            uct_iface_release_desc(rdesc);
        } else {
            free(rdesc);
        }
    }

    if (barrier(oob_sock, progress_worker, if_info.worker)) {
        status = UCS_ERR_IO_ERROR;
    }

out_free_ep:
    uct_ep_destroy(ep);
out_free_ep_addrs:
    free(own_ep);
    free(peer_ep);
out_close_oob_sock:
    close(oob_sock);
out_free_if_addrs:
    free(own_iface);
    free(peer_iface);
out_free_dev_addrs:
    free(own_dev);
    free(peer_dev);
out_destroy_iface:
    uct_iface_close(if_info.iface);
    uct_md_close(if_info.md);
out_destroy_worker:
    uct_worker_destroy(if_info.worker);
out_cleanup_async:
    ucs_async_context_destroy(async);
out:
    return (status == UCS_ERR_UNSUPPORTED) ? UCS_OK : status;
}
