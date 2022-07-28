/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCX_HELLO_WORLD_H
#define UCX_HELLO_WORLD_H

#include <ucs/memory/memory_type.h>

#include <sys/poll.h>
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
    fprintf(stderr, "  -p <port>     Set alternative server port (default:13337)\n");
    fprintf(stderr, "  -6            Use IPv6 address in data exchange\n");
    fprintf(stderr, "  -s <size>     Set test string length (default:16)\n");
    fprintf(stderr, "  -m <mem type> Memory type of messages\n");
    fprintf(stderr, "                host - system memory (default)\n");
    if (check_mem_type_support(UCS_MEMORY_TYPE_CUDA)) {
        fprintf(stderr, "                cuda - NVIDIA GPU memory\n");
    }
    if (check_mem_type_support(UCS_MEMORY_TYPE_CUDA_MANAGED)) {
        fprintf(stderr, "                cuda-managed - NVIDIA GPU managed/unified memory\n");
    }
}

int connect_common(const char *server, uint16_t server_port, sa_family_t af)
{
    int sockfd   = -1;
    int listenfd = -1;
    int optval   = 1;
    char service[8];
    struct addrinfo hints, *res, *t;
    int ret;

    snprintf(service, sizeof(service), "%u", server_port);
    memset(&hints, 0, sizeof(hints));
    hints.ai_flags    = (server == NULL) ? AI_PASSIVE : 0;
    hints.ai_family   = af;
    hints.ai_socktype = SOCK_STREAM;

    ret = getaddrinfo(server, service, &hints, &res);
    CHKERR_JUMP(ret < 0, "getaddrinfo() failed", out);

    for (t = res; t != NULL; t = t->ai_next) {
        sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (sockfd < 0) {
            continue;
        }

        if (server != NULL) {
            if (connect(sockfd, t->ai_addr, t->ai_addrlen) == 0) {
                break;
            }
        } else {
            ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval,
                             sizeof(optval));
            CHKERR_JUMP(ret < 0, "server setsockopt()", err_close_sockfd);

            if (bind(sockfd, t->ai_addr, t->ai_addrlen) == 0) {
                ret = listen(sockfd, 0);
                CHKERR_JUMP(ret < 0, "listen server", err_close_sockfd);

                /* Accept next connection */
                fprintf(stdout, "Waiting for connection...\n");
                listenfd = sockfd;
                sockfd   = accept(listenfd, NULL, NULL);
                close(listenfd);
                break;
            }
        }

        close(sockfd);
        sockfd = -1;
    }

    CHKERR_ACTION(sockfd < 0,
                  (server) ? "open client socket" : "open server socket",
                  (void)sockfd /* no action */);

out_free_res:
    freeaddrinfo(res);
out:
    return sockfd;
err_close_sockfd:
    close(sockfd);
    sockfd = -1;
    goto out_free_res;
}

static inline int
barrier(int oob_sock, void (*progress_cb)(void *arg), void *arg)
{
    struct pollfd pfd;
    int dummy = 0;
    ssize_t res;

    res = send(oob_sock, &dummy, sizeof(dummy), 0);
    if (res < 0) {
        return res;
    }

    pfd.fd      = oob_sock;
    pfd.events  = POLLIN;
    pfd.revents = 0;
    do {
        res = poll(&pfd, 1, 1);
        progress_cb(arg);
    } while (res != 1);

    res = recv(oob_sock, &dummy, sizeof(dummy), MSG_WAITALL);

    /* number of received bytes should be the same as sent */
    return !(res == sizeof(dummy));
}

static inline int generate_test_string(char *str, int size)
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
