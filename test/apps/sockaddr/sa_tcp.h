/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef SA_TCP_H_
#define SA_TCP_H_

#include "sa_base.h"
#include "sa_util.h"


class tcp_socket : public file_desc {
public:
    tcp_socket();

    tcp_socket(int fd);

    virtual ~tcp_socket();

private:
    static int create_socket();
};


class tcp_connection : public connection {
public:
    tcp_connection(const struct sockaddr *addr, socklen_t addrlen);

    tcp_connection(int fd);

private:
    tcp_socket m_socket;
};


class tcp_worker : public worker {
public:
    tcp_worker(const struct sockaddr *listen_addr, socklen_t addrlen);

    virtual conn_ptr_t connect(const struct sockaddr *addr, socklen_t addrlen);

private:
    tcp_socket m_server_socket;
};

#endif
