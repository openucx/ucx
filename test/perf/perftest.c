/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <uct/api/uct.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <netdb.h>
#include <unistd.h>
#include <sys/time.h>


int sock_init(int argc, char **argv, int *my_rank)
{
    struct sockaddr_in inaddr;
    struct hostent *he;
    int sockfd;
    int optval;
    int ret;

    inaddr.sin_port   = htons(12345);
    memset(inaddr.sin_zero, 0, sizeof(inaddr.sin_zero));

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        fprintf(stderr, "socket() failed: %m\n");
        return sockfd;
    }

    if (argc == 1) {
        optval = 1;
        ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
        if (ret < 0) {
            return ret;
        }

        inaddr.sin_family = AF_INET;
        inaddr.sin_addr.s_addr = INADDR_ANY;
        ret = bind(sockfd, (struct sockaddr*)&inaddr, sizeof(inaddr));
        if (ret < 0) {
            fprintf(stderr, "bind() failed: %m\n");
            return ret;
        }

        ret = listen(sockfd, 100);
        if (ret < 0) {
            fprintf(stderr, "listen() failed: %m\n");
            return ret;
        }

        *my_rank = 0;
        printf("Waiting for connection...\n");
        return accept(sockfd, NULL, NULL);
    } else {
         he = gethostbyname(argv[1]);
         if (he == NULL || he->h_addr_list == NULL) {
             fprintf(stderr, "host %s not found: %s\n", argv[1], hstrerror(h_errno));
             return -1;
         }

         inaddr.sin_family = he->h_addrtype;
         memcpy(&inaddr.sin_addr, he->h_addr_list[0], he->h_length);

         ret = connect(sockfd, (struct sockaddr*)&inaddr, sizeof(inaddr));
         if (ret < 0) {
             fprintf(stderr, "connect() failed: %m\n");
             return -1;
         }

         *my_rank = 1;
         return sockfd;
    }
}

ssize_t xchg(int sockfd, void *ptr, size_t length)
{
    if (send(sockfd, ptr, length, 0) != length) {
        fprintf(stderr, "send() failed: %m\n");
        return -1;
    }

    if (recv(sockfd, ptr, length, 0) != length) {
        fprintf(stderr, "recv() failed: %m\n");
        return -1;
    }

    return length;
}

int main(int argc, char **argv)
{
    static const uint64_t count = 1000000ul;
    static volatile uint64_t shared_val8;
    unsigned long vaddr;
    uct_rkey_t rkey;
    void *rkey_buffer;
    uct_lkey_t lkey;
    uct_iface_addr_t *iface_addr;
    uct_ep_addr_t *ep_addr;
    ucs_status_t status;
    uct_context_h context;
    uct_iface_h iface;
    uct_iface_attr_t iface_attr;
    uct_pd_attr_t pd_attr;
    uct_ep_h ep;
    int my_rank;
    int sockfd;
    uint64_t value;
    struct timeval start, end;
    double lat;

    status = uct_init(&context);
    if (status != UCS_OK) {
        fprintf(stderr, "Initialization failed\n");
        return -1;
    }

    status = uct_iface_open(context, "rc_mlx5", "mlx5_0:1", &iface);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to open interface\n");
        return -1;
    }

    status = uct_iface_query(iface, &iface_attr);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to query interface\n");
        return -1;
    }

    status = uct_ep_create(iface, &ep);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to create endpoint\n");
        return -1;
    }

    status = uct_pd_query(iface->pd, &pd_attr);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to query pd\n");
        return -1;
    }

    shared_val8 = -1;
    status = uct_pd_mem_map(iface->pd, (void*)&shared_val8, sizeof(shared_val8), &lkey);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to register\n");
        return -1;
    }

    iface_addr = malloc(iface_attr.iface_addr_len);
    ep_addr    = malloc(iface_attr.ep_addr_len);

    uct_iface_get_address(iface, iface_addr);
    uct_ep_get_address(ep, ep_addr);

    sockfd = sock_init(argc, argv, &my_rank);
    if (sockfd < 0) {
        return -1;
    }

    xchg(sockfd, iface_addr, iface_attr.iface_addr_len);
    xchg(sockfd, ep_addr, iface_attr.ep_addr_len);

    status = uct_ep_connect_to_ep(ep, iface_addr, ep_addr);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to connect to ep\n");
        return -1;
    }

    vaddr = (uintptr_t)&shared_val8;

    rkey_buffer = malloc(pd_attr.rkey_packed_size);
    status = uct_pd_rkey_pack(iface->pd, lkey, rkey_buffer);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to pack rkey\n");
        return -1;
    }

    xchg(sockfd, rkey_buffer, pd_attr.rkey_packed_size);
    status = uct_pd_rkey_unpack(iface->pd, rkey_buffer, &rkey);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to unpack rkey\n");
        return -1;
    }

    shared_val8 = -1;

    xchg(sockfd, &vaddr, sizeof(vaddr));

    free(rkey_buffer);
    free(iface_addr);
    free(ep_addr);
    close(sockfd);

    printf("Starting test...\n");

    value = 0;

    if (my_rank == 0) {
        for (value = 0; value < 100; ++value) {
            uct_ep_put_short(ep, &value, sizeof(value), vaddr, rkey, NULL, NULL);
            while (shared_val8 != value);
        }
    } else if (my_rank == 1) {
        for (value = 0; value < 100; ++value) {
            while (shared_val8 != value);
            uct_ep_put_short(ep, &value, sizeof(value), vaddr, rkey, NULL, NULL);
        }
    }

    gettimeofday(&start, NULL);

    if (my_rank == 0) {
        for (value = 0; value < count; ++value) {
            uct_ep_put_short(ep, &value, sizeof(value), vaddr, rkey, NULL, NULL);
            while (shared_val8 != value);
        }
    } else if (my_rank == 1) {
        for (value = 0; value < count; ++value) {
            while (shared_val8 != value);
            uct_ep_put_short(ep, &value, sizeof(value), vaddr, rkey, NULL, NULL);
        }
    }

    gettimeofday(&end, NULL);

    lat = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec)) / count / 2;

    printf("Test done latency=%.3f usec\n",  lat);

    uct_pd_rkey_release(iface->pd, rkey);
    uct_pd_mem_unmap(iface->pd, lkey);
    uct_ep_destroy(ep);
    uct_iface_close(iface);
    uct_cleanup(context);
    return 0;
}

