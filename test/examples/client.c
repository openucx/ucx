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
 *    ./ucp_hello_world -n <server host name>
 *
 * Notes:
 *
 *    - Client acquires Server UCX address via TCP socket
 *
 *
 * Author:
 *
 *    Ilya Nelkenbaum <ilya@nelkenbaum.com>
 *    Sergey Shalnov <sergeysh@mellanox.com> 7-June-2016
 */

#include "ucx_hello_world.h"

#include <ucp/api/ucp.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <assert.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  /* getopt */
#include <ctype.h>   /* isprint */
#include <pthread.h> /* pthread_self */
#include <errno.h>   /* errno */
#include <time.h>
#include <signal.h>  /* raise */

struct addr_msg {
	uint64_t        data_len;
	int		msg_size;
	char* addr;
};

struct msg {
    uint64_t        data_len;
};

struct ucx_context {
    int             completed;
};

enum ucp_test_mode_t {
    TEST_MODE_PROBE,
    TEST_MODE_WAIT,
    TEST_MODE_EVENTFD
} ucp_test_mode = TEST_MODE_PROBE;

static struct err_handling {
    ucp_err_handling_mode_t ucp_err_mode;
    int                     failure;
} err_handling_opt;

static ucs_status_t client_status = UCS_OK;
static uint16_t server_port = 13337;
static long test_string_length = 16;
static const ucp_tag_t tag  = 0x1337a880u;
static const ucp_tag_t tag_mask = -1;
static ucp_address_t *local_addr;
static ucp_address_t *peer_addr;


int num_conn = 1;
int msg_size = 1024;

static size_t local_addr_len;
static size_t peer_addr_len;

static int parse_cmd(int argc, char * const argv[], char **server_name);

void copy_msg(char* dest , char* str , int size);

static void request_init(void *request)
{
    struct ucx_context *ctx = (struct ucx_context *) request;
    ctx->completed = 0;
}

static void send_handle(void *request, ucs_status_t status)
{
    struct ucx_context *context = (struct ucx_context *) request;

    context->completed = 1;

    printf("[0x%x] send handler called with status %d (%s)\n",
           (unsigned int)pthread_self(), status, ucs_status_string(status));
}

static void failure_handler(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    ucs_status_t *arg_status = (ucs_status_t *)arg;

    printf("[0x%x] failure handler called with status %d (%s)\n",
           (unsigned int)pthread_self(), status, ucs_status_string(status));

    *arg_status = status;
}

static void recv_handle(void *request, ucs_status_t status,
                        ucp_tag_recv_info_t *info)
{
    struct ucx_context *context = (struct ucx_context *) request;

    context->completed = 1;

    printf("[0x%x] receive handler called with status %d (%s), length %lu\n",
           (unsigned int)pthread_self(), status, ucs_status_string(status),
           info->length);
}

static void wait(ucp_worker_h ucp_worker, struct ucx_context *context)
{
    while (context->completed == 0) {
        ucp_worker_progress(ucp_worker);
    }
}

static ucs_status_t test_poll_wait(ucp_worker_h ucp_worker)
{
    int ret = -1, err = 0;
    ucs_status_t status;
    int epoll_fd_local = 0, epoll_fd = 0;
    struct epoll_event ev;
    ev.data.u64 = 0;

    status = ucp_worker_get_efd(ucp_worker, &epoll_fd);
    CHKERR_JUMP(UCS_OK != status, "ucp_worker_get_efd", err);

    /* It is recommended to copy original fd */
    epoll_fd_local = epoll_create(1);

    ev.data.fd = epoll_fd;
    ev.events = EPOLLIN;
    err = epoll_ctl(epoll_fd_local, EPOLL_CTL_ADD, epoll_fd, &ev);
    CHKERR_JUMP(err < 0, "add original socket to the new epoll\n", err_fd);

    /* Need to prepare ucp_worker before epoll_wait */
    status = ucp_worker_arm(ucp_worker);
    if (status == UCS_ERR_BUSY) { /* some events are arrived already */
        ret = UCS_OK;
        goto err_fd;
    }
    CHKERR_JUMP(status != UCS_OK, "ucp_worker_arm\n", err_fd);

    do {
        ret = epoll_wait(epoll_fd_local, &ev, 1, -1);
    } while ((ret == -1) && (errno == EINTR));

    ret = UCS_OK;

err_fd:
    close(epoll_fd_local);

err:
    return ret;
}


static void flush_callback(void *request, ucs_status_t status)
{
}

static ucs_status_t flush_ep(ucp_worker_h worker, ucp_ep_h ep)
{
    void *request;

    request = ucp_ep_flush_nb(ep, 0, flush_callback);
    if (request == NULL) {
        return UCS_OK;
    } else if (UCS_PTR_IS_ERR(request)) {
        return UCS_PTR_STATUS(request);
    } else {
        ucs_status_t status;
        do {
            ucp_worker_progress(worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        ucp_request_release(request);
        return status;
    }
}


int main(int argc, char **argv)
{

	/* UCP temporary vars */
	ucp_params_t ucp_params;
	ucp_worker_params_t worker_params;
	ucp_config_t *config;
	ucs_status_t status;

	/* UCP handler objects */
	ucp_context_h ucp_context;
	ucp_worker_h ucp_worker;

	/* OOB connection vars */
	uint64_t addr_len = 0;
	char *client_target_name = NULL;
	int oob_sock = -1;
	int ret = -1;
	

	memset(&ucp_params, 0, sizeof(ucp_params));
	memset(&worker_params, 0, sizeof(worker_params));

	/* Parse the command line */
	status = parse_cmd(argc, argv, &client_target_name);
	CHKERR_JUMP(status != UCS_OK, "parse_cmd\n", err);

	/* UCP initialization */
	status = ucp_config_read(NULL, NULL, &config);
	CHKERR_JUMP(status != UCS_OK, "ucp_config_read\n", err);

	ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES |
		UCP_PARAM_FIELD_REQUEST_SIZE |
		UCP_PARAM_FIELD_REQUEST_INIT;
	ucp_params.features     = UCP_FEATURE_TAG | UCP_FEATURE_RMA;
	if (ucp_test_mode == TEST_MODE_WAIT || ucp_test_mode == TEST_MODE_EVENTFD) {
		ucp_params.features |= UCP_FEATURE_WAKEUP;
	}
	ucp_params.request_size    = sizeof(struct ucx_context);
	ucp_params.request_init    = request_init;

	status = ucp_init(&ucp_params, config, &ucp_context);


	ucp_config_print(config, stdout, NULL, UCS_CONFIG_PRINT_CONFIG);

	ucp_config_release(config);
	CHKERR_JUMP(status != UCS_OK, "ucp_init\n", err);

	worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
	worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

	status = ucp_worker_create(ucp_context, &worker_params, &ucp_worker);
  CHKERR_JUMP(status != UCS_OK, "ucp_worker_create\n", err_cleanup);

        status = ucp_worker_get_address(ucp_worker, &local_addr, &local_addr_len);
        CHKERR_JUMP(status != UCS_OK, "ucp_worker_get_address\n", err_worker);

        printf("[0x%x] local address length: %lu\n",
                        (unsigned int)pthread_self(), local_addr_len);

        /* OOB connection establishment */
        peer_addr_len = local_addr_len;

        oob_sock = client_connect(client_target_name, server_port);
        CHKERR_JUMP(oob_sock < 0, "client_connect\n", err_addr);

        ret = recv(oob_sock, &addr_len, sizeof(addr_len), 0);
        CHKERR_JUMP(ret < 0, "receive address length\n", err_addr);

        peer_addr_len = addr_len;
        peer_addr = malloc(peer_addr_len);
        CHKERR_JUMP(!peer_addr, "allocate memory\n", err_addr);

        ret = recv(oob_sock, peer_addr, peer_addr_len, 0);
        CHKERR_JUMP(ret < 0, "receive address\n", err_peer_addr);

        ucp_tag_recv_info_t info_tag;
        ucp_tag_message_h msg_tag;
        ucp_ep_h server_ep;
        ucp_ep_params_t ep_params;
        struct msg *msg = 0;
        struct ucx_context *request = 0;
        size_t msg_len = 0;
        struct addr_msg *addr_msg = 0;

      /* Send client UCX address to server */
	ep_params.field_mask      = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
		UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
		UCP_EP_PARAM_FIELD_ERR_HANDLER |
		UCP_EP_PARAM_FIELD_USER_DATA;
	ep_params.address         = peer_addr;
	ep_params.err_mode        = err_handling_opt.ucp_err_mode;
	ep_params.err_handler.cb  = failure_handler;
	ep_params.err_handler.arg = NULL;
	ep_params.user_data       = &client_status;

        status = ucp_ep_create(ucp_worker, &ep_params, &server_ep);
        CHKERR_JUMP(status != UCS_OK, "ucp_ep_create\n", err);

        msg_len = sizeof(*msg) + local_addr_len;
        msg = calloc(1, msg_len);
        CHKERR_JUMP(!msg, "allocate memory\n", err_ep);

        msg->data_len = local_addr_len;
        memcpy(msg + 1, local_addr, local_addr_len);

        request = ucp_tag_send_nb(server_ep, msg, msg_len,
                        ucp_dt_make_contig(1), tag,
                        send_handle);
        if (UCS_PTR_IS_ERR(request)) {
                fprintf(stderr, "unable to send UCX address message\n");
                free(msg);
                goto err_ep;
        } else if (UCS_PTR_STATUS(request) != UCS_OK) {
                wait(ucp_worker, request);
                request->completed = 0; /* Reset request state before recycling it */
                ucp_request_release(request);
        }

        free (msg);

        if (err_handling_opt.failure) {
                fprintf(stderr, "Emulating unexpected failure on client side\n");
                raise(SIGKILL);
        }


	/* Send test string to client */

	//create rkey

        ucp_mem_h memh;
        ucp_mem_map_params_t params;
	size_t rkey_size;
	char* rkey_buf;
	char* buff;

	buff = malloc(1024);
        CHKERR_JUMP(!buff, "allocate memory\n", err_ep);
	int i;
	for( i = 0 ; i < 1024 ; i++) buff[i] = 'a';

	params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.address    = buff;
        params.length     = msg_size;
        //params.flags      = UCP_MEM_MAP_ALLOCATE;

        status = ucp_mem_map(ucp_context, &params, &memh);
	status = ucp_rkey_pack(ucp_context ,memh , (void**)(&rkey_buf) , &rkey_size);
	printf("server va is: %d\n" , params.address);
	printf("server rkey is: %s\n" , rkey_buf);
	//for( i = 0 ; i < rkey_size ; i++) printf("%c" , rkey_buf[i]);
	msg_len = sizeof(*addr_msg) + rkey_size;
	addr_msg = malloc(msg_len);
	CHKERR_JUMP(!addr_msg, "allocate memory\n", err_ep);
	//rkey_msg->rkey = malloc(rkey_size);
	addr_msg->data_len = msg_len - sizeof(struct msg);
	addr_msg->addr = params.address;
	addr_msg->msg_size = msg_size;
	//addr_msg->addr = (uint64_t)params.address;
	printf("addr_msg-> addr: %d\n" , addr_msg->addr);
	//(msg+1) = rkey_buf;
	copy_msg((char*)(addr_msg+1) , rkey_buf , rkey_size);
	request = ucp_tag_send_nb(server_ep, addr_msg, msg_len,
			ucp_dt_make_contig(1), tag,
			send_handle);
	if (UCS_PTR_IS_ERR(request)) {
		fprintf(stderr, "unable to send UCX data message\n");
		free(addr_msg);
		goto err_ep;
	} else if (UCS_PTR_STATUS(request) != UCS_OK) {
		printf("UCX data message was scheduled for send\n");
		wait(ucp_worker, request);
		request->completed = 0;
		ucp_request_release(request);
	}
	msg_tag = NULL;
	do {
		/* Progressing before probe to update the state */
		ucp_worker_progress(ucp_worker);

		/* Probing incoming events in non-block mode */
		msg_tag = ucp_tag_probe_nb(ucp_worker, tag, tag_mask, 1, &info_tag);
	} while (msg_tag == NULL);


	msg = malloc(info_tag.length);
	CHKERR_JUMP(!msg, "allocate memory\n", err);
	request = ucp_tag_msg_recv_nb(ucp_worker, msg, info_tag.length,
			ucp_dt_make_contig(1), msg_tag, recv_handle);

	if (UCS_PTR_IS_ERR(request)) {
		fprintf(stderr, "unable to receive UCX address message (%s)\n",
				ucs_status_string(UCS_PTR_STATUS(request)));
		free(msg);
		goto err;
	} else {
		wait(ucp_worker, request);
		request->completed = 0;
		ucp_request_release(request);
		printf("UCX finish was received\n");
	}

	free(msg);

	status = flush_ep(ucp_worker, server_ep);
	fprintf(stderr, "ucp_ep_flush is completed with status %d (%s)\n",
			status, ucs_status_string(status));
	status = ucp_get(client_ep , (void*)buff , 512 , (uintptr_t)addr_msg->addr , rkey);
	ret = 0;
	free(addr_msg);
        ucp_mem_unmap(ucp_context , memh);
        ucp_rkey_buffer_release((void*)rkey_buf);
err_ep:
	ucp_ep_destroy(server_ep);

	if (!err_handling_opt.failure) {
		/* Make sure remote is disconnected before destroying local worker */
		barrier(oob_sock);
	}
	close(oob_sock);

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

int parse_cmd(int argc, char * const argv[], char **server_name)
{
    int c = 0, index = 0;
    opterr = 0;

    err_handling_opt.ucp_err_mode   = UCP_ERR_HANDLING_MODE_NONE;
    err_handling_opt.failure        = 0;

    while ((c = getopt(argc, argv, "wfbei:n:m:p:s:h")) != -1) {
        switch (c) {
        case 'w':
            ucp_test_mode = TEST_MODE_WAIT;
            break;
        case 'f':
            ucp_test_mode = TEST_MODE_EVENTFD;
            break;
        case 'b':
            ucp_test_mode = TEST_MODE_PROBE;
            break;
        case 'e':
            err_handling_opt.ucp_err_mode   = UCP_ERR_HANDLING_MODE_PEER;
            err_handling_opt.failure        = 1;
            break;
        case 'i':
            *server_name = optarg;
            break;
        case 'n':
            num_conn = atoi(optarg);
            break;
        case 'm':
            msg_size = atoi(optarg);
            break;
        case 'p':
            server_port = atoi(optarg);
            if (server_port <= 0) {
                fprintf(stderr, "Wrong server port number %d\n", server_port);
                return UCS_ERR_UNSUPPORTED;
            }
            break;
        case 's':
            test_string_length = atol(optarg);
            if (test_string_length <= 0) {
                fprintf(stderr, "Wrong string size %ld\n", test_string_length);
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
            fprintf(stderr, "Usage: ucp_hello_world [parameters]\n");
            fprintf(stderr, "UCP hello world client/server example utility\n");
            fprintf(stderr, "\nParameters are:\n");
            fprintf(stderr, "  -w      Select test mode \"wait\" to test "
                    "ucp_worker_wait function\n");
            fprintf(stderr, "  -f      Select test mode \"event fd\" to test "
                    "ucp_worker_get_efd function with later poll\n");
            fprintf(stderr, "  -b      Select test mode \"busy polling\" to test "
                    "ucp_tag_probe_nb and ucp_worker_progress (default)\n");
            fprintf(stderr, "  -e      Emulate unexpected failure on server side"
                    "and handle an error on client side with enabled "
                    "UCP_ERR_HANDLING_MODE_PEER\n");
            fprintf(stderr, "  -n name Set node name or IP address "
                    "of the server (required for client and should be ignored "
                    "for server)\n");
            fprintf(stderr, "  -p port Set alternative server port (default:13337)\n");
            fprintf(stderr, "  -s size Set test string length (default:16)\n");
            fprintf(stderr, "\n");
            return UCS_ERR_UNSUPPORTED;
        }
    }
    fprintf(stderr, "INFO: UCP_HELLO_WORLD mode = %d server = %s port = %d\n",
            ucp_test_mode, *server_name, server_port);

    for (index = optind; index < argc; index++) {
        fprintf(stderr, "WARNING: Non-option argument %s\n", argv[index]);
    }
    return UCS_OK;
}

int run_server()
{
    struct sockaddr_in inaddr;
    int lsock  = -1;
    int dsock  = -1;
    int optval = 1;
    int ret;

    lsock = socket(AF_INET, SOCK_STREAM, 0);
    CHKERR_JUMP(lsock < 0, "open server socket\n", err);

    optval = 1;
    ret = setsockopt(lsock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
    CHKERR_JUMP(ret < 0, "setsockopt server\n", err_sock);

    inaddr.sin_family      = AF_INET;
    inaddr.sin_port        = htons(server_port);
    inaddr.sin_addr.s_addr = INADDR_ANY;
    memset(inaddr.sin_zero, 0, sizeof(inaddr.sin_zero));
    ret = bind(lsock, (struct sockaddr*)&inaddr, sizeof(inaddr));
    CHKERR_JUMP(ret < 0, "bind server\n", err_sock);

    ret = listen(lsock, 0);
    CHKERR_JUMP(ret < 0, "listen server\n", err_sock);

    printf("Waiting for connection...\n");

    /* Accept next connection */
    dsock = accept(lsock, NULL, NULL);
    CHKERR_JUMP(dsock < 0, "accept server\n", err_sock);

    close(lsock);

    return dsock;

err_sock:
    close(lsock);

err:
    return -1;
}

int run_client(const char *server)
{
    struct sockaddr_in conn_addr;
    struct hostent *he;
    int connfd;
    int ret;

    connfd = socket(AF_INET, SOCK_STREAM, 0);
    if (connfd < 0) {
        fprintf(stderr, "socket() failed: %s\n", strerror(errno));
        return -1;
    }

    he = gethostbyname(server);
    CHKERR_JUMP((he == NULL || he->h_addr_list == NULL), "found host\n", err_conn);

    conn_addr.sin_family = he->h_addrtype;
    conn_addr.sin_port   = htons(server_port);

    memcpy(&conn_addr.sin_addr, he->h_addr_list[0], he->h_length);
    memset(conn_addr.sin_zero, 0, sizeof(conn_addr.sin_zero));

    ret = connect(connfd, (struct sockaddr*)&conn_addr, sizeof(conn_addr));
    CHKERR_JUMP(ret < 0, "connect client\n", err_conn);

    return connfd;

err_conn:
    close(connfd);

    return -1;
}





void copy_msg(char* dest , char* str , int size){


	int i;
        for (i = 0; i < (size-1); ++i) {
            dest[i] = str[i];
        }

}
