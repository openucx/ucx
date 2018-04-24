/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "sa_util.h"

#include <sys/epoll.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstring>
#include <climits>


error::error(const std::string& message) : m_message(message) {
}

error::~error() throw() {
}

const char* error::what() const throw() {
    return m_message.c_str();
}

sys_error::~sys_error() throw() {
}

sys_error::sys_error(const std::string& message, int errn) :
    error(message + ": " + strerror(errn) + " (" + std::to_string(errn) + ")") {
}

file_desc::file_desc(int fd) : m_fd(fd) {
}

file_desc::~file_desc() {
    int ret = ::close(m_fd);
    if (ret < 0) {
        fprintf(stderr, "Warning: failed to close fd %d: %m", m_fd);
    }
}

file_desc::operator int() const {
    return m_fd;
}
