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
#include <pthread.h>
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

struct msg {
    uint64_t        data_len;
};

struct addr_msg {
        uint64_t        data_len;
        int		msg_size;
	char* addr;
};

struct ucx_context {
    int             completed;
};

typedef struct wrapper {
	int client_num;
	ucp_context_h context;
}wrapper;


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
//int msg_size = 1024;
//static ucp_address_t *local_addr;
//static ucp_address_t *peer_addr;

//static size_t local_addr_len;
//static size_t peer_addr_len;

void* run_server(void* thread_info);

static int parse_cmd(int argc, char * const argv[], int *server_name);

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

	ucs_status_t status;
	int i , rc , workers_num = 1;
	/* Parse the command line */
	status = parse_cmd(argc, argv, &workers_num);
	CHKERR_JUMP(status != UCS_OK, "parse_cmd\n", err);
	printf("clients num is: %d\n" , workers_num);

	/* UCP temporary vars */
	ucp_params_t ucp_params;
	ucp_config_t *config;

	/* UCP handler objects */
	ucp_context_h ucp_context;

	memset(&ucp_params, 0, sizeof(ucp_params));

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

	wrapper *thread_wrappers;
	pthread_t *client_threads;
	int *nums;
	pthread_t connection_manager_thread;
	thread_wrappers = malloc(workers_num * sizeof(wrapper));
	client_threads = malloc(workers_num * sizeof(pthread_t));
        nums = malloc(workers_num * sizeof(int));
        for(i = 0; i < workers_num; i++){
		nums[i] = i+1;
		thread_wrappers[i].client_num = nums[i];
		thread_wrappers[i].context = ucp_context;
		rc = pthread_create(&client_threads[i] , NULL , &run_server , &thread_wrappers[i]);
		if(rc){
			perror("pthread_create error is:");
			return 1;
		}
	}

        for(i = 0; i < workers_num; i++){
                pthread_join(client_threads[i],NULL);
        }

err:
	ucp_cleanup(ucp_context);
	return 0;
}


void* run_server(void* thread_info){

	wrapper* info = (wrapper*) thread_info;
	ucp_address_t *local_addr;
	ucp_address_t *peer_addr;

	size_t local_addr_len;
	size_t peer_addr_len;

	/* UCP temporary vars */
	ucs_status_t status;
	ucp_worker_params_t worker_params;

	/* UCP handler objects */
	ucp_worker_h ucp_worker;

	/* OOB connection vars */
	uint64_t addr_len = 0;
	int oob_sock = -1;
	int ret = -1;
	
	memset(&worker_params, 0, sizeof(worker_params));

	worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        //worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
        worker_params.thread_mode = UCS_THREAD_MODE_SERIALIZED;

        status = ucp_worker_create(info->context, &worker_params, &ucp_worker);
        CHKERR_JUMP(status != UCS_OK, "ucp_worker_create\n", err_cleanup);

        status = ucp_worker_get_address(ucp_worker, &local_addr, &local_addr_len);
        CHKERR_JUMP(status != UCS_OK, "ucp_worker_get_address\n", err_worker);


        printf("[0x%x] local address length: %lu\n",
                        (unsigned int)pthread_self(), local_addr_len);

        /* OOB connection establishment */
	printf("my port is: %d\n" , server_port + info->client_num);
        oob_sock = server_connect(server_port + info->client_num);
        CHKERR_JUMP(oob_sock < 0, "server_connect\n", err_peer_addr);

        addr_len = local_addr_len;
        ret = send(oob_sock, &addr_len, sizeof(addr_len), 0);
        CHKERR_JUMP((ret < 0 || ret != sizeof(addr_len)),
                        "send address length\n", err_peer_addr);

        ret = send(oob_sock, local_addr, local_addr_len, 0);
        CHKERR_JUMP((ret < 0 || ret != local_addr_len),
                        "send address\n", err_peer_addr);


        ucp_tag_recv_info_t info_tag;
        ucp_tag_message_h msg_tag;
        ucp_ep_h client_ep;
        ucp_ep_params_t ep_params;
        struct msg *msg = 0;
        struct ucx_context *request = 0;
        size_t msg_len = 0;
        struct rkey_msg *rkey_msg = 0;
             ucp_rkey_h rkey;
        struct addr_msg *addr_msg = 0;


	/* Receive client UCX address */
        do {
                /* Progressing before probe to update the state */
                ucp_worker_progress(ucp_worker);

                /* Probing incoming events in non-block mode */
                msg_tag = ucp_tag_probe_nb(ucp_worker, tag, tag_mask, 1, &info_tag);
        } while (msg_tag == NULL);


	printf("Im waiting for %d\n" , info->client_num);
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
                printf("UCX address message was received\n");
        }

        peer_addr = malloc(msg->data_len);
        if (!peer_addr) {
                fprintf(stderr, "unable to allocate memory for peer address\n");
                free(msg);
                goto err;
        }

        peer_addr_len = msg->data_len;
        memcpy(peer_addr, msg + 1, peer_addr_len);

        free(msg);

	/* Receive test string from server */
	        ep_params.field_mask      = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                UCP_EP_PARAM_FIELD_ERR_HANDLER |
                UCP_EP_PARAM_FIELD_USER_DATA;
        ep_params.address         = peer_addr;
        ep_params.err_mode        = err_handling_opt.ucp_err_mode;
        ep_params.err_handler.cb  = failure_handler;
        ep_params.err_handler.arg = NULL;
        ep_params.user_data       = &client_status;

        status = ucp_ep_create(ucp_worker, &ep_params, &client_ep);
        CHKERR_JUMP(status != UCS_OK, "ucp_ep_create\n", err);

	for (;;) {

		/* Probing incoming events in non-block mode */
		msg_tag = ucp_tag_probe_nb(ucp_worker, tag, tag_mask, 1, &info_tag);
		if (msg_tag != NULL) {
			/* Message arrived */
			break;
		} else if (ucp_worker_progress(ucp_worker)) {
			/* Some events were polled; try again without going to sleep */
			continue;
		}

		/* If we got here, ucp_worker_progress() returned 0, so we can sleep.
		 * Following blocked methods used to polling internal file descriptor
		 * to make CPU idle and don't spin loop
		 */
		if (ucp_test_mode == TEST_MODE_WAIT) {
			/* Polling incoming events*/
			status = ucp_worker_wait(ucp_worker);
			CHKERR_JUMP(status != UCS_OK, "ucp_worker_wait\n", err_ep);
		} else if (ucp_test_mode == TEST_MODE_EVENTFD) {
			status = test_poll_wait(ucp_worker);
			CHKERR_JUMP(status != UCS_OK, "test_poll_wait\n", err_ep);
		}
	}

	addr_msg = malloc(info_tag.length);
	CHKERR_JUMP(!addr_msg, "allocate memory\n", err_ep);

	request = ucp_tag_msg_recv_nb(ucp_worker, addr_msg, info_tag.length,
			ucp_dt_make_contig(1), msg_tag,
			recv_handle);

	if (UCS_PTR_IS_ERR(request)) {
		fprintf(stderr, "unable to receive UCX data message (%u)\n",
				UCS_PTR_STATUS(request));
		free(addr_msg);
		goto err_ep;
	} else {
		wait(ucp_worker, request);
		request->completed = 0;
		ucp_request_release(request);
		printf("UCX data message was received\n");
	}

	printf("\n\n----- UCP TEST SUCCESS ----\n\n");
	printf("rkey is: %s\n", (char *)(addr_msg + 1));
	printf("addr is: %d\n", addr_msg->addr);
	printf("msg_size is: %d\n", addr_msg->msg_size);
	printf("\n\n---------------------------\n\n");
	int msg_size = addr_msg->msg_size;
	status = ucp_ep_rkey_unpack(client_ep , (void*)(addr_msg+1) , &rkey);
	//char* rptr;
	//status = ucp_rkey_ptr(rkey , addr_msg->addr , (void**)(&rptr));	
	char* buff;
	buff = malloc(msg_size);
	//buff = malloc(1024);
	printf("buff before is: %s\n" , buff);
	CHKERR_JUMP(!buff, "allocate memory\n", err_ep);
	//status = ucp_get(server_ep , (void*)buff , 512 , (uint64_t)rptr , rkey);
	status = ucp_get(client_ep , (void*)buff , msg_size , (uintptr_t)addr_msg->addr , rkey);
	//status = ucp_get(client_ep , (void*)buff , 512 , (uintptr_t)addr_msg->addr , rkey);
	printf("buff after is: %s\n" , buff);
	free(addr_msg);


	msg_len = sizeof(*msg);
	msg = malloc(msg_len);
	CHKERR_JUMP(!msg, "allocate memory\n", err_ep);

	msg->data_len = 0;

	request = ucp_tag_send_nb(client_ep, msg, msg_len,
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

	ret = 0;
	
	free (buff);
err_ep:
	ucp_ep_destroy(client_ep);


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


err:
	return NULL;
}

int parse_cmd(int argc, char * const argv[], int *workers_num)
{
    int c = 0, index = 0;
    opterr = 0;

    err_handling_opt.ucp_err_mode   = UCP_ERR_HANDLING_MODE_NONE;
    err_handling_opt.failure        = 0;

    while ((c = getopt(argc, argv, "wfben:m:p:s:h")) != -1) {
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
        case 'n':
            *workers_num = atoi(optarg);
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
            fprintf(stderr, "Usage: ucp_rdma_server [parameters]\n");
            fprintf(stderr, "UCP get server example utility\n");
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
            fprintf(stderr, "  -n	Set the total clients number\n");
            fprintf(stderr, "  -m	Set the message size\n");
            fprintf(stderr, "  -p port Set alternative server port (default:13337)\n");
            fprintf(stderr, "  -s size Set test string length (default:16)\n");
            fprintf(stderr, "\n");
            return UCS_ERR_UNSUPPORTED;
        }
    }
    fprintf(stderr, "INFO: UCP_RDMA_SERVER mode = %d server = null port = %d\n",
            ucp_test_mode, server_port);

    for (index = optind; index < argc; index++) {
        fprintf(stderr, "WARNING: Non-option argument %s\n", argv[index]);
    }
    return UCS_OK;
}

