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

tcp_connection::tcp_connection(const struct sockaddr *addr, socklen_t addrlen) :
                m_is_closed(false) {
    initialize();
    int ret = ::connect(m_socket, addr, addrlen);
    if ((ret < 0) && (errno != EINPROGRESS)) {
        throw sys_error("failed to connect tcp socket", errno);
    }
}

tcp_connection::tcp_connection(int fd) : m_socket(fd), m_is_closed(false) {
    initialize();
}

void tcp_connection::initialize() {
    int ret = fcntl(m_socket, F_SETFL, fcntl(m_socket, F_GETFL) | O_NONBLOCK);
    if (ret < 0) {
        throw sys_error("failed to set tcp socket to nonblocking", errno);
    }

    set_id(m_socket);
}

void tcp_connection::add_to_evpoll(evpoll_set& evpoll) {
    evpoll.add(m_socket, EPOLLIN | EPOLLOUT | EPOLLERR | EPOLLET);
}

size_t tcp_connection::send(const char *buffer, size_t size) {
    ssize_t ret = ::send(m_socket, buffer, size, 0);
    if (ret < 0) {
        if (errno != EAGAIN) {
            throw sys_error("failed to send on tcp socket", errno);
        }
        return 0;
    }
    return ret;
}

size_t tcp_connection::recv(char *buffer, size_t size) {
    ssize_t ret = ::recv(m_socket, buffer, size, 0);
    if (ret < 0) {
        if (errno != EAGAIN) {
            throw sys_error("failed to receive from tcp socket", errno);
        }
        return 0;
    }
    if (ret == 0) {
        m_is_closed = true;
    }
    return ret;
}

bool tcp_connection::is_closed() const {
    return m_is_closed;
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

void tcp_worker::add_to_evpoll(evpoll_set& evpoll) {
    evpoll.add(m_server_socket, EPOLLIN | EPOLLERR);
}

void tcp_worker::wait(const evpoll_set& evpoll, conn_handler_t conn_handler,
                      data_handler_t data_handler, int timeout_ms) {
    std::vector<evpoll_set::event> events;
    evpoll.wait(events, timeout_ms);
    for (auto ev : events) {
        if (ev.fd == m_server_socket) {
            int ret = accept(m_server_socket, NULL, NULL);
            if (ret < 0) {
                throw sys_error("failed to accept", errno);
            }
            auto conn = std::make_shared<tcp_connection>(ret);
            conn_handler(conn);
        } else {
            data_handler(ev.fd, ev.ev_flags);
        }
    }
}

std::shared_ptr<connection> tcp_worker::connect(const struct sockaddr *addr,
                                                socklen_t addrlen) {
    return std::make_shared<tcp_connection>(addr, addrlen);
}
