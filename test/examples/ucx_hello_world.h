/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCX_HELLO_WORLD_H
#define UCX_HELLO_WORLD_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <netdb.h>

#define CHKERR_JUMP(_cond, _msg, _label)            \
do {                                                \
    if (_cond) {                                    \
        fprintf(stderr, "Failed to %s\n", _msg);    \
        goto _label;                                \
    }                                               \
} while (0)

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

static void barrier(int oob_sock)
{
    int dummy = 0;
    send(oob_sock, &dummy, sizeof(dummy), 0);
    recv(oob_sock, &dummy, sizeof(dummy), 0);
}

static void generate_random_string(char *str, int size)
{
    int i;
    static int init = 0;
    /* randomize seed only once */
    if (!init) {
        srand(time(NULL));
        init = 1;
    }

    for (i = 0; i < (size-1); ++i) {
        str[i] =  'A' + (rand() % 26);
    }
    str[i] = 0;
}

#endif /* UCX_HELLO_WORLD_H */
