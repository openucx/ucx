/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_helpers.h"

#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>
#include <ucs/time/time.h>
#include <ucs/sys/string.h>

namespace ucs {

int test_time_multiplier()
{
    int factor = 1;
#if _BullseyeCoverage
    factor *= 10;
#endif
    if (RUNNING_ON_VALGRIND) {
        factor *= 20;
    }
    return factor;
}

std::ostream& operator<<(std::ostream& os, const std::vector<char>& vec) {
    static const size_t LIMIT = 100;
    size_t i = 0;
    for (std::vector<char>::const_iterator iter = vec.begin(); iter != vec.end(); ++iter) {
        if (i >= LIMIT) {
            os << "...";
            break;
        }
        int n = static_cast<unsigned char>(*iter);
        os << "[" << i << "]=" << n << " ";
        ++i;
    }
    return os << std::endl;
}

void fill_random(void *data, size_t size)
{
    if (ucs::test_time_multiplier() > 1) {
        memset(data, 0, size);
        return;
    }

    uint64_t seed = rand();
    for (size_t i = 0; i < size / sizeof(uint64_t); ++i) {
        ((uint64_t*)data)[i] = seed;
        seed = seed * 10 + 17;
    }
    size_t remainder = size % sizeof(uint64_t);
    memset((char*)data + size - remainder, 0xab, remainder);
}

scoped_setenv::scoped_setenv(const char *name, const char *value) : m_name(name) {
    if (getenv(name)) {
        m_old_value = getenv(name);
    }
    setenv(m_name.c_str(), value, 1);
}

scoped_setenv::~scoped_setenv() {
    if (!m_old_value.empty()) {
        setenv(m_name.c_str(), m_old_value.c_str(), 1);
    } else {
        unsetenv(m_name.c_str());
    }
}

void safe_usleep(double usec) {
    ucs_time_t current_time = ucs_get_time();
    ucs_time_t end_time = current_time + ucs_time_from_usec(usec);

    while (current_time < end_time) {
        usleep((long)ucs_time_to_usec(end_time - current_time));
        current_time = ucs_get_time();
    }
}

std::string get_iface_ip(const struct sockaddr *ifa_addr) {
    size_t ip_len = ucs_max(INET_ADDRSTRLEN, INET6_ADDRSTRLEN);
    char ip_str[ip_len];

    return ucs_sockaddr_str(ifa_addr, ip_str, ip_len);
}

bool is_inet_addr(const struct sockaddr* ifa_addr) {
    return ((ifa_addr->sa_family == AF_INET) ||
            (ifa_addr->sa_family == AF_INET6));
}

bool is_ib_netdev(const char *ifa_name) {
    char path[PATH_MAX];
    DIR *dir;

    snprintf(path, PATH_MAX, "/sys/class/net/%s/device/infiniband", ifa_name);

    dir = opendir(path);
    if (dir == NULL) {
        return false;
    } else {
        closedir(dir);
        return true;
    }
}

namespace detail {

message_stream::message_stream(const std::string& title) {
    static const char PADDING[] = "          ";
    static const size_t WIDTH = strlen(PADDING);

    msg <<  "[";
    msg.write(PADDING, ucs_max(WIDTH - 1, title.length()) - title.length());
    msg << title << " ] ";
}

message_stream::~message_stream() {
    msg << std::endl;
    std::cout << msg.str() << std::flush;
}

} // detail

} // ucs
