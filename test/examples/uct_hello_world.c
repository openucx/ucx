/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "ucx_hello_world.h"

#include <uct/api/uct.h>

#include <assert.h>
#include <ctype.h>

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
    func_am_t           func_am_type;
    const char         *dev_name;
    const char         *tl_name;
    long                test_strlen;
} cmd_args_t;

typedef struct {
    uct_iface_attr_t    attr;   /* Interface attributes: capabilities and limitations */
    uct_iface_h         iface;  /* Communication interface context */
    uct_md_h            pd;     /* Memory domain */
    uct_worker_h        worker; /* Workers represent allocated resources in a communication thread */
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
    memcpy(dest, bc_args->data, bc_args->len);
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
    return (len >= 0) ? UCS_OK : len;
}

/* Completion callback for am_zcopy */
void zcopy_completion_cb(uct_completion_t *self, ucs_status_t status)
{
    zcopy_comp_t *comp = (zcopy_comp_t *)self;
    assert((comp->uct_comp.count == 0) && (status == UCS_OK));
    uct_md_mem_dereg(comp->md, comp->memh);
    desc_holder = (void *)0xDEADBEEF;
}

ucs_status_t do_am_zcopy(iface_info_t *if_info, uct_ep_h ep, uint8_t id,
                         const cmd_args_t *cmd_args, char *buf)
{
    uct_mem_h memh;
    uct_iov_t iov;
    zcopy_comp_t comp;

    ucs_status_t status = uct_md_mem_reg(if_info->pd, buf, cmd_args->test_strlen,
                                         UCT_MD_MEM_ACCESS_RMA, &memh);
    iov.buffer          = buf;
    iov.length          = cmd_args->test_strlen;
    iov.memh            = memh;
    iov.stride          = 0;
    iov.count           = 1;

    comp.uct_comp.func  = zcopy_completion_cb;
    comp.uct_comp.count = 1;
    comp.md             = if_info->pd;
    comp.memh           = memh;

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
                          const char *remote_str)
{
    fprintf(stdout, "\n\n----- UCT TEST SUCCESS ----\n\n");
    fprintf(stdout, "[%s] %s sent %s", label, local_str, remote_str);
    fprintf(stdout, "\n\n---------------------------\n");
    fflush(stdout);
}

/* Callback to handle receive active message */
static ucs_status_t hello_world(void *arg, void *data, size_t length, unsigned flags)
{
    recv_desc_t *rdesc;
    func_am_t func_am_type = *(func_am_t *)arg;
    print_strings("callback", func_am_t_str(func_am_type), data);

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
    rdesc->is_uct_desc = 0;
    memcpy(rdesc + 1, data, length);
    desc_holder = rdesc;
    return UCS_OK;
}

/* init the transport  by its name */
static ucs_status_t init_iface(char *dev_name, char *tl_name,
                               func_am_t func_am_type,
                               iface_info_t *iface_p)
{
    ucs_status_t        status;
    uct_iface_config_t  *config; /* Defines interface configuration options */
    uct_iface_params_t  params;

    params.open_mode            = UCT_IFACE_OPEN_MODE_DEVICE;
    params.mode.device.tl_name  = tl_name;
    params.mode.device.dev_name = dev_name;
    params.stats_root           = NULL;
    params.rx_headroom          = sizeof(recv_desc_t);

    UCS_CPU_ZERO(&params.cpu_mask);
    /* Read transport-specific interface configuration */
    status = uct_md_iface_config_read(iface_p->pd, tl_name, NULL, NULL, &config);
    CHKERR_JUMP(UCS_OK != status, "setup iface_config", error_ret);

    /* Open communication interface */
    status = uct_iface_open(iface_p->pd, iface_p->worker, &params, config,
                            &iface_p->iface);
    uct_config_release(config);
    CHKERR_JUMP(UCS_OK != status, "open temporary interface", error_ret);

    /* Enable progress on the interface */
    uct_iface_progress_enable(iface_p->iface,
                              UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    /* Get interface attributes */
    status = uct_iface_query(iface_p->iface, &iface_p->attr);
    CHKERR_JUMP(UCS_OK != status, "query iface", error_iface);

    /* Check if current device and transport support required active messages */
    if ((func_am_type == FUNC_AM_SHORT) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_AM_SHORT)) {
        return UCS_OK;
    }

    if ((func_am_type == FUNC_AM_BCOPY) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_AM_BCOPY)) {
        return UCS_OK;
    }

    if ((func_am_type == FUNC_AM_ZCOPY) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_AM_ZCOPY)) {
        return UCS_OK;
    }

error_iface:
    uct_iface_close(iface_p->iface);
error_ret:
    return UCS_ERR_UNSUPPORTED;
}

/* Device and transport to be used are determined by minimum latency */
static ucs_status_t dev_tl_lookup(const cmd_args_t *cmd_args,
                                  iface_info_t *iface_p)
{
    uct_md_resource_desc_t  *md_resources; /* Memory domain resource descriptor */
    uct_tl_resource_desc_t  *tl_resources; /*Communication resource descriptor */
    unsigned                num_md_resources; /* Number of protected domain */
    unsigned                num_tl_resources; /* Number of transport resources resource objects created */
    uct_md_config_t         *md_config;
    ucs_status_t            status;
    int                     i;
    int                     j;

    status = uct_query_md_resources(&md_resources, &num_md_resources);
    CHKERR_JUMP(UCS_OK != status, "query for memory domain resources", error_ret);

    /* Iterate through protected domain resources */
    for (i = 0; i < num_md_resources; ++i) {
        status = uct_md_config_read(md_resources[i].md_name, NULL, NULL, &md_config);
        CHKERR_JUMP(UCS_OK != status, "read PD config", release_pd);

        status = uct_md_open(md_resources[i].md_name, md_config, &iface_p->pd);
        uct_config_release(md_config);
        CHKERR_JUMP(UCS_OK != status, "open memory domains", release_pd);

        status = uct_md_query_tl_resources(iface_p->pd, &tl_resources, &num_tl_resources);
        CHKERR_JUMP(UCS_OK != status, "query transport resources", close_pd);

        /* Go through each available transport and find the proper name */
        for (j = 0; j < num_tl_resources; ++j) {
            if (!strcmp(cmd_args->dev_name, tl_resources[j].dev_name) &&
                !strcmp(cmd_args->tl_name, tl_resources[j].tl_name)) {
                status = init_iface(tl_resources[j].dev_name,
                                    tl_resources[j].tl_name,
                                    cmd_args->func_am_type, iface_p);
                if (UCS_OK == status) {
                    fprintf(stdout, "Using %s with %s.\n",
                            tl_resources[j].dev_name,
                            tl_resources[j].tl_name);
                    fflush(stdout);
                    uct_release_tl_resource_list(tl_resources);
                    goto release_pd;
                }
            }
        }
        uct_release_tl_resource_list(tl_resources);
        uct_md_close(iface_p->pd);
    }

    fprintf(stderr, "No supported (dev/tl) found (%s/%s)\n",
            cmd_args->dev_name, cmd_args->tl_name);
    status = UCS_ERR_UNSUPPORTED;

release_pd:
    uct_release_md_resource_list(md_resources);
error_ret:
    return status;
close_pd:
    uct_md_close(iface_p->pd);
    goto release_pd;
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
    fprintf(stderr, "  -d      Select device name\n");
    fprintf(stderr, "  -t      Select transport layer\n");
    fprintf(stderr, "  -n name Set node name or IP address "
            "of the server (required for client and should be ignored "
            "for server)\n");
    fprintf(stderr, "  -p port Set alternative server port (default:13337)\n");
    fprintf(stderr, "  -s size Set test string length (default:16)\n");
    fprintf(stderr, "\n");
    return UCS_ERR_UNSUPPORTED;
}

int parse_cmd(int argc, char * const argv[], cmd_args_t *args)
{
    int c = 0, index = 0;

    assert(args);
    memset(args, 0, sizeof(*args));

    /* Defaults */
    args->server_port   = 13337;
    args->func_am_type  = FUNC_AM_SHORT;
    args->test_strlen   = 16;

    opterr = 0;
    while ((c = getopt(argc, argv, "ibzd:t:n:p:s:h")) != -1) {
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
            if (args->test_strlen <= 0) {
                fprintf(stderr, "Wrong string size %ld\n", args->test_strlen);
                return UCS_ERR_UNSUPPORTED;
            }
            break;
        case '?':
            if (optopt == 's') {
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            } else if (isprint (optopt)) {
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            } else {
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            }
        case 'h':
        default:
            return print_err_usage();
        }
    }
    fprintf(stderr, "INFO: UCT_HELLO_WORLD AM function = %s server = %s port = %d\n",
            func_am_t_str(args->func_am_type), args->server_name,
            args->server_port);

    for (index = optind; index < argc; index++) {
        fprintf(stderr, "WARNING: Non-option argument %s\n", argv[index]);
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
    if ((ret < 0) || (ret != slen)) {
        fprintf(stderr, "failed to send buffer\n");
        return -1;
    }

    ret = recv(sock, &rlen, sizeof(rlen), 0);
    if (ret < 0) {
        fprintf(stderr, "failed to receive device address length\n");
        return -1;
    }

    *rbuf = calloc(1, rlen);
    if (!*rbuf) {
        fprintf(stderr, "failed to allocate receive buffer\n");
        return -1;
    }

    ret = recv(sock, *rbuf, rlen, 0);
    if (ret < 0) {
        fprintf(stderr, "failed to receive device address\n");
        return -1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    uct_device_addr_t   *own_dev;
    uct_device_addr_t   *peer_dev   = NULL;
    uct_iface_addr_t    *own_iface;
    uct_iface_addr_t    *peer_iface = NULL;
    uct_ep_addr_t       *own_ep;
    uct_ep_addr_t       *peer_ep    = NULL;
    ucs_status_t        status      = UCS_OK; /* status codes for UCS */
    uct_ep_h            ep;                   /* Remote endpoint */
    ucs_async_context_t *async;               /* Async event context manages
                                                 times and fd notifications */
    cmd_args_t          cmd_args;

    iface_info_t        if_info;
    uint8_t             id = 0;
    int                 oob_sock = -1;  /* OOB connection socket */

    /* Parse the command line */
    if (parse_cmd(argc, argv, &cmd_args)) {
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    /* Initialize context
     * It is better to use different contexts for different workers
     */
    status = ucs_async_context_create(UCS_ASYNC_MODE_THREAD, &async);
    CHKERR_JUMP(UCS_OK != status, "init async context", out);

    /* Create a worker object */
    status = uct_worker_create(async, UCS_THREAD_MODE_SINGLE, &if_info.worker);
    CHKERR_JUMP(UCS_OK != status, "create worker", out_cleanup_async);

    /* Search for the desired transport */
    status = dev_tl_lookup(&cmd_args, &if_info);
    CHKERR_JUMP(UCS_OK != status, "find supported device and transport",
                out_destroy_worker);

    own_dev = (uct_device_addr_t*)calloc(1, if_info.attr.device_addr_len);
    CHKERR_JUMP(NULL == own_dev, "allocate memory for dev addr",
                out_destroy_iface);

    own_iface = (uct_iface_addr_t*)calloc(1, if_info.attr.iface_addr_len);
    CHKERR_JUMP(NULL == own_iface, "allocate memory for if addr",
                out_free_dev_addrs);

    /* Get device address */
    status = uct_iface_get_device_address(if_info.iface, own_dev);
    CHKERR_JUMP(UCS_OK != status, "get device address", out_free_if_addrs);

    if (cmd_args.server_name) {
        oob_sock = client_connect(cmd_args.server_name, cmd_args.server_port);
        if (oob_sock < 0) {
            goto out_free_if_addrs;
        }
    } else {
        oob_sock = server_connect(cmd_args.server_port);
        if (oob_sock < 0) {
            goto out_free_if_addrs;
        }
    }

    status = sendrecv(oob_sock, own_dev, if_info.attr.device_addr_len,
                      (void **)&peer_dev);
    CHKERR_JUMP(0 != status, "device exchange", out_free_dev_addrs);

    status = uct_iface_is_reachable(if_info.iface, peer_dev, NULL);
    CHKERR_JUMP(0 == status, "reach the peer", out_free_if_addrs);

    /* Get interface address */
    if (if_info.attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        status = uct_iface_get_address(if_info.iface, own_iface);
        CHKERR_JUMP(UCS_OK != status, "get interface address", out_free_if_addrs);

        status = sendrecv(oob_sock, own_iface, if_info.attr.iface_addr_len,
                          (void **)&peer_iface);
        CHKERR_JUMP(0 != status, "ifaces exchange", out_free_if_addrs);
    }

    if (if_info.attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        own_ep = (uct_ep_addr_t*)calloc(1, if_info.attr.ep_addr_len);
        CHKERR_JUMP(NULL == own_ep, "allocate memory for ep addrs", out_free_if_addrs);

        /* Create new endpoint */
        status = uct_ep_create(if_info.iface, &ep);
        CHKERR_JUMP(UCS_OK != status, "create endpoint", out_free_ep_addrs);

        /* Get endpoint address */
        status = uct_ep_get_address(ep, own_ep);
        CHKERR_JUMP(UCS_OK != status, "get endpoint address", out_free_ep);

        status = sendrecv(oob_sock, own_ep, if_info.attr.ep_addr_len,
                          (void **)&peer_ep);
        CHKERR_JUMP(0 != status, "EPs exchange", out_free_ep);

        /* Connect endpoint to a remote endpoint */
        status = uct_ep_connect_to_ep(ep, peer_dev, peer_ep);
        barrier(oob_sock);
    } else if (if_info.attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        /* Create an endpoint which is connected to a remote interface */
        status = uct_ep_create_connected(if_info.iface, peer_dev, peer_iface, &ep);
    } else {
        status = UCS_ERR_UNSUPPORTED;
    }
    CHKERR_JUMP(UCS_OK != status, "connect endpoint", out_free_ep);

    if (cmd_args.test_strlen > func_am_max_size(cmd_args.func_am_type, &if_info.attr)) {
        status = UCS_ERR_UNSUPPORTED;
        fprintf(stderr, "Test string is too long: %ld, max supported: %lu\n",
                cmd_args.test_strlen,
                func_am_max_size(cmd_args.func_am_type, &if_info.attr));
        goto out_free_ep;
    }

    /*Set active message handler */
    status = uct_iface_set_am_handler(if_info.iface, id, hello_world,
                                      &cmd_args.func_am_type,
                                      UCT_CB_FLAG_SYNC);
    CHKERR_JUMP(UCS_OK != status, "set callback", out_free_ep);

    if (cmd_args.server_name) {
        char *str = (char *)malloc(cmd_args.test_strlen);
        generate_random_string(str, cmd_args.test_strlen);

        /* Send active message to remote endpoint */
        if (cmd_args.func_am_type == FUNC_AM_SHORT) {
            status = do_am_short(&if_info, ep, id, &cmd_args, str);
        } else if (cmd_args.func_am_type == FUNC_AM_BCOPY) {
            status = do_am_bcopy(&if_info, ep, id, &cmd_args, str);
        } else if (cmd_args.func_am_type == FUNC_AM_ZCOPY) {
            status = do_am_zcopy(&if_info, ep, id, &cmd_args, str);
        }

        free(str);
        CHKERR_JUMP(UCS_OK != status, "send active msg", out_free_ep);
    } else {
        recv_desc_t *rdesc;

        while (!desc_holder) {
            /* Explicitly progress any outstanding active message requests */
            uct_worker_progress(if_info.worker);
        }

        rdesc = desc_holder;
        print_strings("main", func_am_t_str(cmd_args.func_am_type),
                                            (char *)(rdesc + 1));
        if (rdesc->is_uct_desc) {
            /* Release descriptor because callback returns UCS_INPROGRESS */
            uct_iface_release_desc(rdesc);
        } else {
            free(rdesc);
        }
    }

    barrier(oob_sock);
    close(oob_sock);

out_free_ep:
    uct_ep_destroy(ep);
out_free_ep_addrs:
    free(own_ep);
    free(peer_ep);
out_free_if_addrs:
    free(own_iface);
    free(peer_iface);
out_free_dev_addrs:
    free(own_dev);
    free(peer_dev);
out_destroy_iface:
    uct_iface_close(if_info.iface);
    uct_md_close(if_info.pd);
out_destroy_worker:
    uct_worker_destroy(if_info.worker);
out_cleanup_async:
    ucs_async_context_destroy(async);
out:
    return status == UCS_ERR_UNSUPPORTED ? UCS_OK : status;
}
