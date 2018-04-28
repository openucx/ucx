/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "sa_tcp.h"

#include <sys/socket.h>
#include <sys/epoll.h>
#include <sys/fcntl.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>


tcp_socket::tcp_socket() : file_desc(create_socket()) {
}

tcp_socket::tcp_socket(int fd) : file_desc(fd) {
}

tcp_socket::~tcp_socket() {
}

int tcp_socket::create_socket() {
    int fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd < 0) {
        throw sys_error("failed to create tcp socket", errno);
    }
    return fd;
}

tcp_connection::tcp_connection(const struct sockaddr *addr, socklen_t addrlen) {
    set_id(m_socket);
    int ret = ::connect(m_socket, addr, addrlen);
    if (ret < 0) {
        throw sys_error("failed to connect tcp socket", errno);
    }
}

tcp_connection::tcp_connection(int fd) : m_socket(fd) {
    set_id(m_socket);
}

tcp_worker::tcp_worker(const struct sockaddr *listen_addr, socklen_t addrlen) {
    int retb = ::bind(m_server_socket, listen_addr, addrlen);
    if (retb != 0) {
        throw sys_error("failed to bind tcp socket", errno);
    }

    int retl = ::listen(m_server_socket, 1024);
    if (retl != 0) {
        throw sys_error("failed to listen on tcp socket", errno);
    }
}

std::shared_ptr<connection> tcp_worker::connect(const struct sockaddr *addr,
                                                socklen_t addrlen) {
    return std::make_shared<tcp_connection>(addr, addrlen);
}
