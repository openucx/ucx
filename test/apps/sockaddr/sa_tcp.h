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

    virtual void add_to_evpoll(evpoll_set& evpoll);

    virtual size_t send(const char *buffer, size_t size);

    virtual size_t recv(char *buffer, size_t size);

    virtual bool is_closed() const;

private:
    void initialize();

    tcp_socket m_socket;
    bool       m_is_closed;
};


class tcp_worker : public worker {
public:
    tcp_worker(const struct sockaddr *listen_addr, socklen_t addrlen);

    virtual void add_to_evpoll(evpoll_set& evpoll);

    virtual conn_ptr_t connect(const struct sockaddr *addr, socklen_t addrlen);

    virtual void wait(const evpoll_set& evpoll, conn_handler_t conn_handler,
                      data_handler_t data_handler, int timeout_ms);

private:
    tcp_socket m_server_socket;
};

#endif
