/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <ucs/type/status.h>
#include <ucs/async/async.h>
#include <uct/api/uct.h>
#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#define CHKERR_JUMP(cond, msg, label)       \
do {                                        \
    if (cond) {                             \
    fprintf(stderr, "Failed to %s\n", msg); \
    goto label;                             \
    }                                       \
} while (0)

static int holder = 1;

struct iface_info {
    /* Interface attributes: capabilities and limitations */
    uct_iface_attr_t attr;

    /* Communication interface context */
    uct_iface_h iface;

    /* Protection domain */
    uct_pd_h pd;

    /* Workers represent allocated resources in a communication thread */
    uct_worker_h worker;
};

/* Callback for active message */
static ucs_status_t hello_world(void *arg, void *data, size_t length, void *desc)
{
    printf("Hello World!!!\n");fflush(stdout);
    holder = 0;

    return UCS_OK;
}

/* init the transport  by its name */
static ucs_status_t init_iface(char *dev_name, char *tl_name, struct iface_info *iface_p)
{
    ucs_status_t status;
    uct_iface_config_t *config; /* Defines interface configuration options */

    /* Read transport-specific interface configuration */
    status = uct_iface_config_read(tl_name, NULL, NULL, &config);
    CHKERR_JUMP(UCS_OK != status, "setup iface_config", error_ret);

    /* Open communication interface */
    status = uct_iface_open(iface_p->pd, iface_p->worker, tl_name, dev_name, 0, config,
            &iface_p->iface);
    uct_config_release(config);
    CHKERR_JUMP(UCS_OK != status, "open temporary interface", error_ret);

    /* Get interface attributes */
    status = uct_iface_query(iface_p->iface, &iface_p->attr);
    CHKERR_JUMP(UCS_OK != status, "query iface", error_iface);

    /* Check if current device and transport support short active messages */
    if (iface_p->attr.cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
        return UCS_OK;
    }

error_iface:
    uct_iface_close(iface_p->iface);
error_ret:
    return UCS_ERR_UNSUPPORTED;
}

/* Device and transport to be used are determined by minimum latency */
static ucs_status_t dev_tl_lookup(const char *dev_name, const char *tl_name, struct iface_info *iface_p)
{
    int i;
    int j;
    ucs_status_t status;
    uct_pd_resource_desc_t *pd_resources; /* Protection domain resource descriptor */
    uct_tl_resource_desc_t *tl_resources; /*Communication resource descriptor */
    unsigned num_pd_resources; /* Number of protected domain */
    unsigned num_tl_resources; /* Number of transport resources resource objects created */
    uct_pd_config_t *pd_config;

    status = uct_query_pd_resources(&pd_resources, &num_pd_resources);
    CHKERR_JUMP(UCS_OK != status, "query for protected domain resources", error_ret);

    /* Iterate through protected domain resources */
    for (i = 0; i < num_pd_resources; ++i) {
        status = uct_pd_config_read(pd_resources[i].pd_name, NULL, NULL, &pd_config);
        CHKERR_JUMP(UCS_OK != status, "read PD config", release_pd);

        status = uct_pd_open(pd_resources[i].pd_name, pd_config, &iface_p->pd);
        uct_config_release(pd_config);
        CHKERR_JUMP(UCS_OK != status, "open protected domains", release_pd);

        status = uct_pd_query_tl_resources(iface_p->pd, &tl_resources, &num_tl_resources);
        CHKERR_JUMP(UCS_OK != status, "query transport resources", close_pd);

        /* Go through each available transport and find the proper name */
        for (j = 0; j < num_tl_resources; ++j) {
            if (!strcmp(dev_name, tl_resources[j].dev_name) &&
                !strcmp(tl_name, tl_resources[j].tl_name)) {
                status = init_iface(tl_resources[j].dev_name, tl_resources[j].tl_name, iface_p);
                if (UCS_OK == status) {
                    printf("Using %s with %s.\n", tl_resources[j].dev_name, tl_resources[j].tl_name);
                    fflush(stdout);
                    uct_release_tl_resource_list(tl_resources);
                    goto release_pd;
                }
            }
        }
        uct_release_tl_resource_list(tl_resources);
        uct_pd_close(iface_p->pd);
    }

    fprintf(stderr, "No supported (dev/tl) found (%s/%s)\n", dev_name, tl_name);
    status = UCS_ERR_UNSUPPORTED;

release_pd:
    uct_release_pd_resource_list(pd_resources);
error_ret:
    return status;
close_pd:
    uct_pd_close(iface_p->pd);
    goto release_pd;
}

int main(int argc, char **argv)
{
    /* MPI is initially used to swap the endpoint and interface addresses so each
     * process has knowledge of the others. */
    int partner;
    int size, rank;
    struct sockaddr *ep_addrs;    /* Endpoint addresses (own and peer's) */
    struct sockaddr *iface_addrs; /* Interface addresses (own and peer's) */
    ucs_status_t status;          /* status codes for UCS */
    uct_ep_h ep;                  /* Remote endpoint */
    ucs_async_context_t async;    /* Async event context manages times and fd notifications */
    uint8_t id = 0;
    void *arg;
    const char *tl_name = NULL;
    const char *dev_name = NULL;
    struct iface_info if_info;
    int exit_fail = 1;

    optind = 1;
    if (3 == argc) {
        dev_name = argv[1];
        tl_name  = argv[2];
    } else {
        printf("Usage: %s (<dev-name> <tl-name>)\n", argv[0]);
        fflush(stdout);
        return 1;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        fprintf(stderr, "Failed to create enough mpi processes\n");
        goto out;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (0 == rank) {
        partner = 1;
    } else if (1 == rank) {
        partner = 0;
    } else {
        /* just wait for other processes in MPI_Finalize */
        exit_fail = 0;
        goto out;
    }

    /* Initialize context */
    status = ucs_async_context_init(&async, UCS_ASYNC_MODE_THREAD);
    CHKERR_JUMP(UCS_OK != status, "init async context", out);

    /* Create a worker object */
    status = uct_worker_create(&async, UCS_THREAD_MODE_SINGLE, &if_info.worker);
    CHKERR_JUMP(UCS_OK != status, "create worker", out_cleanup_async);

    /* Search for the desired transport */
    status = dev_tl_lookup(dev_name, tl_name, &if_info);
    CHKERR_JUMP(UCS_OK != status, "find supported device and transport", out_destroy_worker);

    iface_addrs = (struct sockaddr*) calloc(2, if_info.attr.iface_addr_len);
    CHKERR_JUMP(NULL == iface_addrs, "allocate memory for if addrs", out_destroy_iface);

    /* Get interface address */
    status = uct_iface_get_address(if_info.iface, iface_addrs);
    CHKERR_JUMP(UCS_OK != status, "get interface address", out_free_if_addrs);

    /* Expect that addr len is the same on both peers.
     * Store peer's address in iface_addrs[1] */
    MPI_Sendrecv(iface_addrs, if_info.attr.iface_addr_len, MPI_BYTE, partner, 0,
                 &iface_addrs[1], if_info.attr.iface_addr_len, MPI_BYTE, partner,
                 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    status = uct_iface_is_reachable(if_info.iface, &iface_addrs[1]);
    CHKERR_JUMP(0 == status, "reach the peer", out_free_if_addrs);

    ep_addrs = (struct sockaddr*)calloc(2, if_info.attr.ep_addr_len);
    CHKERR_JUMP(NULL == ep_addrs, "allocate memory for ep addrs", out_free_if_addrs);

    if (if_info.attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        /* Create new endpoint */
        status = uct_ep_create(if_info.iface, &ep);
        CHKERR_JUMP(UCS_OK != status, "create endpoint", out_free_ep_addrs);

        /* Get endpoint address */
        status = uct_ep_get_address(ep, ep_addrs);
        CHKERR_JUMP(UCS_OK != status, "get endpoint address", out_free_ep);
    }

    /* Again, expect that ep addr len is the same on both peers.
     * Store peer's address in ep_addrs[1] */
    MPI_Sendrecv(ep_addrs, if_info.attr.ep_addr_len, MPI_BYTE, partner, 0,
                 &ep_addrs[1], if_info.attr.ep_addr_len, MPI_BYTE, partner,
                 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (if_info.attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        /* Connect endpoint to a remote endpoint */
        status = uct_ep_connect_to_ep(ep, &ep_addrs[1]);
        MPI_Barrier(MPI_COMM_WORLD);
    } else if (if_info.attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        /* Create an endpoint which is connected to a remote interface */
        status = uct_ep_create_connected(if_info.iface, &iface_addrs[1], &ep);
    } else {
        status = UCS_ERR_UNSUPPORTED;
    }
    CHKERR_JUMP(UCS_OK != status, "connect endpoint", out_free_ep);

    /*Set active message handler */
    status = uct_iface_set_am_handler(if_info.iface, id, hello_world, arg, UCT_AM_CB_FLAG_SYNC);
    CHKERR_JUMP(UCS_OK != status, "set callback", out_free_ep);

    if (0 == rank) {
        uint64_t header;
        char payload[8];
        unsigned length = sizeof(payload);
        /* Send active message to remote endpoint */
        status = uct_ep_am_short(ep, id, header, payload, length);
        CHKERR_JUMP(UCS_OK != status, "send active msg", out_free_ep);
    } else if (1 == rank) {
        while (holder) {
            /* Explicitly progress any outstanding active message requests */
            uct_worker_progress(if_info.worker);
        }
    }

    /* Everything is fine, we need to call MPI_Finalize rather than MPI_Abort */
    exit_fail = 0;

out_free_ep:
    uct_ep_destroy(ep);
out_free_ep_addrs:
    free(ep_addrs);
out_free_if_addrs:
    free(iface_addrs);
out_destroy_iface:
    uct_iface_close(if_info.iface);
    uct_pd_close(if_info.pd);
out_destroy_worker:
    uct_worker_destroy(if_info.worker);
out_cleanup_async:
    ucs_async_context_cleanup(&async);
out:
    (0 == exit_fail) ? MPI_Finalize() : MPI_Abort(MPI_COMM_WORLD, 1);
    return exit_fail;
}
