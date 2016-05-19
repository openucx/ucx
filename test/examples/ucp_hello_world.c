/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#define HAVE_CONFIG_H /* Force using config.h, so test would fail if header
                         actually tries to use it */

/*
 * UCP hello world client / server example utility
 * -----------------------------------------------
 *
 * Server side:
 *
 *    ./ucp_hello_world
 *
 * Client side:
 *
 *    ./ucp_hello_world    <server host name>
 *
 * Notes:
 *
 *    - Client acquires Server UCX address via TCP socket
 *
 *
 * Author:
 *
 *    Ilya Nelkenbaum <ilya@nelkenbaum.com>
 *
 */

#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <assert.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct msg {
    uint64_t        data_len;
    uint8_t         data[0];
};

struct ucx_context {
    int             completed;
};


static uint16_t server_port = 13337;
static const ucp_tag_t tag  = 0x1337a880u;

static ucp_address_t *local_addr;
static ucp_address_t *peer_addr;

static size_t local_addr_len;
static size_t peer_addr_len;

static char *test_str = "Hello UCP World!!!!";


static void request_init(void *request)
{
    struct ucx_context *ctx = (struct ucx_context *) request;
    ctx->completed = 0;
}

static int run_server()
{
    struct sockaddr_in inaddr;
    int lsock  = -1;
    int dsock  = -1;
    int optval = 1;
    int ret;

    lsock = socket(AF_INET, SOCK_STREAM, 0);
    if (lsock < 0) {
        fprintf(stderr, "server socket() failed\n");
        goto err;
    }

    optval = 1;
    ret = setsockopt(lsock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
    if (ret < 0) {
        fprintf(stderr, "server setsockopt() failed\n");
        goto err_sock;
    }

    inaddr.sin_family      = AF_INET;
    inaddr.sin_port        = htons(server_port);
    inaddr.sin_addr.s_addr = INADDR_ANY;
    memset(inaddr.sin_zero, 0, sizeof(inaddr.sin_zero));
    ret = bind(lsock, (struct sockaddr*)&inaddr, sizeof(inaddr));
    if (ret < 0) {
        fprintf(stderr, "server bind() failed\n");
        goto err_sock;
    }

    ret = listen(lsock, 0);
    if (ret < 0) {
        fprintf(stderr, "server listen() failed\n");
        goto err_sock;
    }

    printf("Waiting for connection...\n");

    /* Accept next connection */
    dsock = accept(lsock, NULL, NULL);
    if (dsock < 0) {
        fprintf(stderr, "server accept() failed\n");
        goto err_sock;
    }

    close(lsock);

    return dsock;

err_sock:
    close(lsock);

err:
    return -1;
}

static int run_client(const char *server)
{
    struct sockaddr_in conn_addr;
    struct hostent *he;
    int connfd;
    int ret;

    connfd = socket(AF_INET, SOCK_STREAM, 0);
    if (connfd < 0) {
        fprintf(stderr, "socket() failed: %m\n");
        return -1;
    }

    he = gethostbyname(server);
    if (he == NULL || he->h_addr_list == NULL) {
        fprintf(stderr, "host %s not found: %s\n", server, hstrerror(h_errno));
        goto err_conn;
    }

    conn_addr.sin_family = he->h_addrtype;
    conn_addr.sin_port   = htons(server_port);

    memcpy(&conn_addr.sin_addr, he->h_addr_list[0], he->h_length);
    memset(conn_addr.sin_zero, 0, sizeof(conn_addr.sin_zero));

    ret = connect(connfd, (struct sockaddr*)&conn_addr, sizeof(conn_addr));
    if (ret < 0) {
        fprintf(stderr, "connect() failed: %m\n");
        goto err_conn;
    }

    return connfd;

err_conn:
    close(connfd);

    return -1;
}

static void send_handle(void *request, ucs_status_t status)
{
    struct ucx_context *context = (struct ucx_context *) request;

    context->completed = 1;

    printf("[0x%x] send handler called with status %d\n",
           (unsigned int)pthread_self(), status);
}

static void recv_handle(void *request, ucs_status_t status,
                        ucp_tag_recv_info_t *info)
{
    struct ucx_context *context = (struct ucx_context *) request;

    context->completed = 1;

    printf("[0x%x] receive handler called with status %d (length %u)\n",
           (unsigned int)pthread_self(), status, info->length);
}

static void wait(ucp_worker_h *ucp_worker, struct ucx_context *context)
{
    while (context->completed == 0)
        ucp_worker_progress(*ucp_worker);
}

static int run_ucx_client(ucp_worker_h *ucp_worker)
{
    ucp_tag_recv_info_t info_tag;
    ucp_tag_message_h msg_tag;
    ucs_status_t status;
    ucp_ep_h server_ep;
    struct msg *msg;
    struct ucx_context *request;
    size_t msg_len;
    int ret = -1;

    /* Send client UCX address to server */
    status = ucp_ep_create(*ucp_worker, peer_addr, &server_ep);
    if (status != UCS_OK) {
        goto err;
    }

    msg_len = sizeof(*msg) + local_addr_len;
    msg = calloc(1, msg_len);
    if (!msg) {
        goto err_ep;
    }

    msg->data_len = local_addr_len;
    memcpy(msg->data, local_addr, local_addr_len);

    request = ucp_tag_send_nb(server_ep, msg, msg_len,
                              ucp_dt_make_contig(1), tag,
                              send_handle);
    if (UCS_PTR_IS_ERR(request)) {
        fprintf(stderr, "unable to send UCX address message\n");
        free(msg);
        goto err_ep;
    } else if (UCS_PTR_STATUS(request) != UCS_OK) {
        printf("UCX address message was scheduled for send\n");
        wait(ucp_worker, request);
        ucp_request_release(request);
    }

    free (msg);

    /* Receive test string from server */
    for ( ; ; ucp_worker_progress(*ucp_worker)) {
        msg_tag = ucp_tag_probe_nb(*ucp_worker, tag, (ucp_tag_t)-1, 1,
                                   &info_tag);
        if (msg_tag == NULL) {
            continue;
        }

        msg = malloc(info_tag.length);
        if (!msg) {
            fprintf(stderr, "unable to allocate memory\n");
            goto err_ep;
        }

        request = ucp_tag_msg_recv_nb(*ucp_worker, msg, info_tag.length,
                                      ucp_dt_make_contig(1), msg_tag,
                                      recv_handle);
        if (UCS_PTR_IS_ERR(request)) {
            fprintf(stderr, "unable to receive UCX data message (%u)\n",
                    UCS_PTR_STATUS(request));
            free(msg);
            goto err_ep;
        } else {
            wait(ucp_worker, request);
            ucp_request_release(request);
            printf("UCX data message was received\n");
        }

        break;
    }

    printf("\n\n----- UCP TEST SUCCESS ----\n\n");
    printf("%s", msg->data);
    printf("\n\n---------------------------\n\n");

    free(msg);

    ret = 0;

err_ep:
    ucp_ep_destroy(server_ep);

err:
    return ret;
}

static int run_ucx_server(ucp_worker_h *ucp_worker)
{
    ucp_tag_recv_info_t info_tag;
    ucp_tag_message_h msg_tag;
    ucs_status_t status;
    ucp_ep_h client_ep;
    struct msg *msg;
    struct ucx_context *request;
    size_t msg_len;
    int ret = -1;

    /* Receive client UCX address */
    for ( ; ; ucp_worker_progress(*ucp_worker)) {
        msg_tag = ucp_tag_probe_nb(*ucp_worker, tag, (ucp_tag_t)-1, 1,
                                   &info_tag);
        if (msg_tag == NULL) {
            continue;
        }

        msg = malloc(info_tag.length);
        if (!msg) {
            fprintf(stderr, "unable to allocate memory\n");
            goto err;
        }

        request = ucp_tag_msg_recv_nb(*ucp_worker, msg, info_tag.length,
                                      ucp_dt_make_contig(1), msg_tag,
                                      recv_handle);
        if (UCS_PTR_IS_ERR(request)) {
            fprintf(stderr, "unable to receive UCX address message (%u)\n",
                    UCS_PTR_STATUS(request));
            free(msg);
            goto err;
        } else {
            wait(ucp_worker, request);
            ucp_request_release(request);
            printf("UCX address message was received\n");
        }

        break;
    }


    peer_addr = malloc(msg->data_len);
    if (!peer_addr) {
        fprintf(stderr, "unable to allocate memory for peer address\n");
        free(msg);
        goto err;
    }

    peer_addr_len = msg->data_len;
    memcpy(peer_addr, msg->data, peer_addr_len);

    free(msg);

    /* Send test string to client */
    status = ucp_ep_create(*ucp_worker, peer_addr, &client_ep);
    if (status != UCS_OK) {
        goto err;
    }

    msg_len = sizeof(*msg) + strlen(test_str) + 1;
    msg = calloc(1, msg_len);
    if (!msg) {
        printf("unable to allocate memory\n");
        goto err_ep;
    }

    msg->data_len = msg_len - sizeof(*msg);;
    snprintf(msg->data, msg->data_len, "%s", test_str);

    request = ucp_tag_send_nb(client_ep, msg, msg_len,
                              ucp_dt_make_contig(1), tag,
                              send_handle);
    if (UCS_PTR_IS_ERR(request)) {
        fprintf(stderr, "unable to send UCX data message\n");
        free(msg);
        goto err_ep;
    } else if (UCS_PTR_STATUS(request) != UCS_OK) {
        printf("UCX data message was scheduled for send\n");
        wait(ucp_worker, request);
        ucp_request_release(request);
    }

    ret = 0;
    free(msg);

err_ep:
    ucp_ep_destroy(client_ep);

err:
    return ret;
}

static int run_test(const char *server, ucp_worker_h *ucp_worker)
{
    if (server != NULL) {
        /* client */
        return run_ucx_client(ucp_worker);
    } else {
        /* server */
        return run_ucx_server(ucp_worker);
    }
}

int main(int argc, char **argv)
{
    /* UCP temporary vars */
    ucp_params_t ucp_params;
    ucp_config_t *config;
    ucs_status_t status;

    /* UCP handler objects */
    ucp_context_h ucp_context;
    ucp_worker_h ucp_worker;

    /* OOB connection vars */
    uint64_t addr_len;
    char *server = NULL;
    int oob_sock = -1;

    int ret = -1;

    if (argc >= 2) {
        server = argv[1];
    }

    /* UCP initialization */
    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK) {
        goto err;
    }

    ucp_params.features        = UCP_FEATURE_TAG;
    ucp_params.request_size    = sizeof(struct ucx_context);
    ucp_params.request_init    = request_init;
    ucp_params.request_cleanup = NULL;

    status = ucp_init(&ucp_params, config, &ucp_context);

    ucp_config_print(config, stdout, NULL, UCS_CONFIG_PRINT_CONFIG);

    ucp_config_release(config);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucp_worker_create(ucp_context, UCS_THREAD_MODE_SINGLE,
                               &ucp_worker);
    if (status != UCS_OK) {
        goto err_cleanup;
    }

    status = ucp_worker_get_address(ucp_worker, &local_addr, &local_addr_len);
    if (status != UCS_OK) {
        goto err_worker;
    }

    printf("[0x%x] local address length: %u\n",
           (unsigned int)pthread_self(), local_addr_len);

    /* OOB connection establishment */
    if (server != NULL) {
        /* client */
        peer_addr_len = local_addr_len;

        oob_sock = run_client(server);
        if (oob_sock < 0) {
            goto err_addr;
        }

        ret = recv(oob_sock, &addr_len, sizeof(addr_len), 0);
        if (ret < 0) {
            fprintf(stderr, "failed to receive address length\n");
            goto err_addr;
        }

        peer_addr_len = addr_len;
        peer_addr = malloc(peer_addr_len);
        if (!peer_addr) {
            fprintf(stderr, "unable to allocate memory\n");
            goto err_addr;
        }

        ret = recv(oob_sock, peer_addr, peer_addr_len, 0);
        if (ret < 0) {
            fprintf(stderr, "failed to receive address\n");
            goto err_peer_addr;
        }
    } else {
        /* server */
        oob_sock = run_server();
        if (oob_sock < 0) {
            goto err_peer_addr;
        }

        addr_len = local_addr_len;
        ret = send(oob_sock, &addr_len, sizeof(addr_len), 0);
        if (ret < 0 || ret != sizeof(addr_len)) {
            fprintf(stderr, "failed to send address length\n");
            goto err_peer_addr;
        }

        ret = send(oob_sock, local_addr, local_addr_len, 0);
        if (ret < 0 || ret != local_addr_len) {
            fprintf(stderr, "failed to send address\n");
            goto err_peer_addr;
        }
    }

    close(oob_sock);

    ret = run_test(server, &ucp_worker);

err_peer_addr:
    free(peer_addr);

err_addr:
    ucp_worker_release_address(ucp_worker, local_addr);

err_worker:
    ucp_worker_destroy(ucp_worker);

err_cleanup:
    ucp_cleanup(ucp_context);

err:

    return ret;
}
