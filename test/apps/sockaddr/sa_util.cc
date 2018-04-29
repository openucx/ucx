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

evpoll_set::evpoll_set() : file_desc(create_epfd()) {
}

void evpoll_set::add(int fd, uint32_t ev_flags) {
    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.events  = ev_flags;
    ev.data.fd = fd;
    int ret = ::epoll_ctl(*this, EPOLL_CTL_ADD, fd, &ev);
    if (ret != 0) {
        throw sys_error("failed to add fd to epoll", errno);
    }
}

void evpoll_set::wait(std::vector<event>& events, int timeout_ms) const {
    static const size_t maxevents = 32;
    struct epoll_event ev_array[maxevents];

    LOG_DEBUG << "epoll_wait with timeout " << timeout_ms << " milliseconds";
    int ret = epoll_wait(*this, ev_array, maxevents, timeout_ms);
    if (ret < 0) {
        if (errno != EINTR) {
            throw sys_error("epoll_wait failed", errno);
        }
    } else {
        for (int i = 0; i < ret; ++i) {
            event ev = { ev_array[i].data.fd, ev_array[i].events };
            events.push_back(ev);
        }
    }
}

int evpoll_set::create_epfd() {
    int fd = epoll_create(1);
    if (fd < 0) {
        throw sys_error("failed to create epoll set", errno);
    }
    return fd;
}

log::level_t log::m_log_level = INFO;

log::log(log::level_t level, const std::string& file, int line) :
            m_enabled(level <= m_log_level) {
    if (m_enabled) {
        struct timeval tv;
        gettimeofday(&tv, NULL);

        char cstr[64];
        snprintf(cstr, sizeof(cstr), "[%lu.%06lu] %12s:%-5d",
                 tv.tv_sec, tv.tv_usec, basename(file.c_str()), line);
        m_msg << cstr << " " << level_str(level) << "   ";
    }
}

log::~log() {
    if (m_enabled) {
        m_msg << std::endl;
        std::cout << m_msg.str() << std::flush;
    }
}

std::string log::level_str(log::level_t level) {
    switch (level) {
    case INFO:
        return "INFO ";
    case DEBUG:
        return "DEBUG";
    default:
        throw error("invalid log level");
    }
}

void log::more_verbose() {
    if (m_log_level == INFO) {
        m_log_level = DEBUG;
    }
}
