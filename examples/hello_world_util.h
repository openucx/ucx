/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCX_HELLO_WORLD_H
#define UCX_HELLO_WORLD_H

#include <ucs/memory/memory_type.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <netdb.h>

#ifdef HAVE_CUDA
#  include <cuda.h>
#  include <cuda_runtime.h>
#endif


#define CHKERR_ACTION(_cond, _msg, _action) \
    do { \
        if (_cond) { \
            fprintf(stderr, "Failed to %s\n", _msg); \
            _action; \
        } \
    } while (0)


#define CHKERR_JUMP(_cond, _msg, _label) \
    CHKERR_ACTION(_cond, _msg, goto _label)


#define CHKERR_JUMP_RETVAL(_cond, _msg, _label, _retval) \
    do { \
        if (_cond) { \
            fprintf(stderr, "Failed to %s, return value %d\n", _msg, _retval); \
            goto _label; \
        } \
    } while (0)


static ucs_memory_type_t test_mem_type = UCS_MEMORY_TYPE_HOST;


#define CUDA_FUNC(_func)                                   \
    do {                                                   \
        cudaError_t _result = (_func);                     \
        if (cudaSuccess != _result) {                      \
            fprintf(stderr, "%s failed: %s\n",             \
                    #_func, cudaGetErrorString(_result));  \
        }                                                  \
    } while(0)


void print_common_help(void);

void *mem_type_malloc(size_t length)
{
    void *ptr;

    switch (test_mem_type) {
    case UCS_MEMORY_TYPE_HOST:
        ptr = malloc(length);
        break;
#ifdef HAVE_CUDA
    case UCS_MEMORY_TYPE_CUDA:
        CUDA_FUNC(cudaMalloc(&ptr, length));
        break;
    case UCS_MEMORY_TYPE_CUDA_MANAGED:
        CUDA_FUNC(cudaMallocManaged(&ptr, length, cudaMemAttachGlobal));
        break;
#endif
    default:
        fprintf(stderr, "Unsupported memory type: %d\n", test_mem_type);
        ptr = NULL;
        break;
    }

    return ptr;
}

void mem_type_free(void *address)
{
    switch (test_mem_type) {
    case UCS_MEMORY_TYPE_HOST:
        free(address);
        break;
#ifdef HAVE_CUDA
    case UCS_MEMORY_TYPE_CUDA:
    case UCS_MEMORY_TYPE_CUDA_MANAGED:
        CUDA_FUNC(cudaFree(address));
        break;
#endif
    default:
        fprintf(stderr, "Unsupported memory type: %d\n", test_mem_type);
        break;
    }
}

void *mem_type_memcpy(void *dst, const void *src, size_t count)
{
    switch (test_mem_type) {
    case UCS_MEMORY_TYPE_HOST:
        memcpy(dst, src, count);
        break;
#ifdef HAVE_CUDA
    case UCS_MEMORY_TYPE_CUDA:
    case UCS_MEMORY_TYPE_CUDA_MANAGED:
        CUDA_FUNC(cudaMemcpy(dst, src, count, cudaMemcpyDefault));
        break;
#endif
    default:
        fprintf(stderr, "Unsupported memory type: %d\n", test_mem_type);
        break;
    }

    return dst;
}

void *mem_type_memset(void *dst, int value, size_t count)
{
    switch (test_mem_type) {
    case UCS_MEMORY_TYPE_HOST:
        memset(dst, value, count);
        break;
#ifdef HAVE_CUDA
    case UCS_MEMORY_TYPE_CUDA:
    case UCS_MEMORY_TYPE_CUDA_MANAGED:
        CUDA_FUNC(cudaMemset(dst, value, count));
        break;
#endif
    default:
        fprintf(stderr, "Unsupported memory type: %d", test_mem_type);
        break;
    }

    return dst;
}

int check_mem_type_support(ucs_memory_type_t mem_type)
{
    switch (test_mem_type) {
    case UCS_MEMORY_TYPE_HOST:
        return 1;
    case UCS_MEMORY_TYPE_CUDA:
    case UCS_MEMORY_TYPE_CUDA_MANAGED:
#ifdef HAVE_CUDA
        return 1;
#else
        return 0;
#endif
    default:
        fprintf(stderr, "Unsupported memory type: %d", test_mem_type);
        break;
    }

    return 0;
}

ucs_memory_type_t parse_mem_type(const char *opt_arg)
{
    if (!strcmp(opt_arg, "host")) {
        return UCS_MEMORY_TYPE_HOST;
    } else if (!strcmp(opt_arg, "cuda") &&
               check_mem_type_support(UCS_MEMORY_TYPE_CUDA)) {
        return UCS_MEMORY_TYPE_CUDA;
    } else if (!strcmp(opt_arg, "cuda-managed") &&
               check_mem_type_support(UCS_MEMORY_TYPE_CUDA_MANAGED)) {
        return UCS_MEMORY_TYPE_CUDA_MANAGED;
    } else {
        fprintf(stderr, "Unsupported memory type: \"%s\".\n", opt_arg);
    }

    return UCS_MEMORY_TYPE_LAST;
}

void print_common_help()
{
    fprintf(stderr, "  -n name Set node name or IP address "
            "of the server (required for client and should be ignored "
            "for server)\n");
    fprintf(stderr, "  -p port Set alternative server port (default:13337)\n");
    fprintf(stderr, "  -s size Set test string length (default:16)\n");
    fprintf(stderr, "  -m <mem type>  memory type of messages\n");
    fprintf(stderr, "                 host - system memory (default)\n");
    if (check_mem_type_support(UCS_MEMORY_TYPE_CUDA)) {
        fprintf(stderr, "                 cuda - NVIDIA GPU memory\n");
    }
    if (check_mem_type_support(UCS_MEMORY_TYPE_CUDA_MANAGED)) {
        fprintf(stderr, "                 cuda-managed - NVIDIA GPU managed/unified memory\n");
    }
}

int server_connect(uint16_t server_port)
{
    struct sockaddr_in inaddr;
    int lsock  = -1;
    int dsock  = -1;
    int optval = 1;
    int ret;

    lsock = socket(AF_INET, SOCK_STREAM, 0);
    CHKERR_JUMP(lsock < 0, "open server socket", err);

    optval = 1;
    ret = setsockopt(lsock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
    CHKERR_JUMP(ret < 0, "server setsockopt()", err_sock);

    inaddr.sin_family      = AF_INET;
    inaddr.sin_port        = htons(server_port);
    inaddr.sin_addr.s_addr = INADDR_ANY;
    memset(inaddr.sin_zero, 0, sizeof(inaddr.sin_zero));
    ret = bind(lsock, (struct sockaddr*)&inaddr, sizeof(inaddr));
    CHKERR_JUMP(ret < 0, "bind server", err_sock);

    ret = listen(lsock, 0);
    CHKERR_JUMP(ret < 0, "listen server", err_sock);

    fprintf(stdout, "Waiting for connection...\n");

    /* Accept next connection */
    dsock = accept(lsock, NULL, NULL);
    CHKERR_JUMP(dsock < 0, "accept server", err_sock);

    close(lsock);

    return dsock;

err_sock:
    close(lsock);

err:
    return -1;
}

int client_connect(const char *server, uint16_t server_port)
{
    struct sockaddr_in conn_addr;
    struct hostent *he;
    int connfd;
    int ret;

    connfd = socket(AF_INET, SOCK_STREAM, 0);
    CHKERR_JUMP(connfd < 0, "open client socket", err);

    he = gethostbyname(server);
    CHKERR_JUMP((he == NULL || he->h_addr_list == NULL), "found a host", err_conn);

    conn_addr.sin_family = he->h_addrtype;
    conn_addr.sin_port   = htons(server_port);

    memcpy(&conn_addr.sin_addr, he->h_addr_list[0], he->h_length);
    memset(conn_addr.sin_zero, 0, sizeof(conn_addr.sin_zero));

    ret = connect(connfd, (struct sockaddr*)&conn_addr, sizeof(conn_addr));
    CHKERR_JUMP(ret < 0, "connect client", err_conn);

    return connfd;

err_conn:
    close(connfd);
err:
    return -1;
}

static int barrier(int oob_sock)
{
    int dummy = 0;
    ssize_t res;

    res = send(oob_sock, &dummy, sizeof(dummy), 0);
    if (res < 0) {
        return res;
    }

    res = recv(oob_sock, &dummy, sizeof(dummy), MSG_WAITALL);

    /* number of received bytes should be the same as sent */
    return !(res == sizeof(dummy));
}

static int generate_test_string(char *str, int size)
{
    char *tmp_str;
    int i;

    tmp_str = calloc(1, size);
    CHKERR_ACTION(tmp_str == NULL, "allocate memory\n", return -1);

    for (i = 0; i < (size - 1); ++i) {
        tmp_str[i] = 'A' + (i % 26);
    }

    mem_type_memcpy(str, tmp_str, size);

    free(tmp_str);
    return 0;
}

#endif /* UCX_HELLO_WORLD_H */
