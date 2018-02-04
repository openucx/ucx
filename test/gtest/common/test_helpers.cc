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

void safe_sleep(double sec) {
    ucs_time_t current_time = ucs_get_time();
    ucs_time_t end_time = current_time + ucs_time_from_sec(sec);

    while (current_time < end_time) {
        usleep((long)ucs_time_to_usec(end_time - current_time));
        current_time = ucs_get_time();
    }
}

void safe_usleep(double usec) {
    safe_sleep(usec * 1e-6);
}

bool is_inet_addr(const struct sockaddr* ifa_addr) {
    return ifa_addr->sa_family == AF_INET;
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

uint16_t get_port() {
    int sock_fd, ret;
    ucs_status_t status;
    struct sockaddr_in addr_in, ret_addr;
    socklen_t len = sizeof(ret_addr);
    uint16_t port;

    status = ucs_tcpip_socket_create(&sock_fd);
    EXPECT_EQ(status, UCS_OK);

    memset(&addr_in, 0, sizeof(struct sockaddr_in));
    addr_in.sin_family      = AF_INET;
    addr_in.sin_addr.s_addr = INADDR_ANY;

    do {
        addr_in.sin_port        = htons(0);
        /* Ports below 1024 are considered "privileged" (can be used only by user root).
         * Ports above and including 1024 can be used by anyone */
        ret = bind(sock_fd, (struct sockaddr*)&addr_in, sizeof(struct sockaddr_in));
    } while (ret);

    ret = getsockname(sock_fd, (struct sockaddr*)&ret_addr, &len);
    EXPECT_EQ(ret, 0);
    EXPECT_LT(1023, ntohs(ret_addr.sin_port)) ;

    port = ret_addr.sin_port;
    close(sock_fd);
    return port;
}

void *mmap_fixed_address() {
    return (void*)0xff0000000;
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

namespace ucp {


data_type_desc_t &
data_type_desc_t::make(ucp_datatype_t datatype, const void *buf, size_t length,
                       size_t iov_cnt)
{
    EXPECT_FALSE(is_valid());

    if (m_length == 0) {
        m_length = length;
    }

    if (m_origin == uintptr_t(NULL)) {
        m_origin = uintptr_t(buf);
    }

    m_dt = datatype;
    memset(m_iov, 0, sizeof(m_iov));

    switch (m_dt & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        m_buf   = buf;
        m_count = length / ucp_contig_dt_elem_size(datatype);
        break;
    case UCP_DATATYPE_IOV:
    {
        const size_t iov_length = (length > iov_cnt) ?
            ucs::rand() % (length / iov_cnt) : 0;
        size_t iov_length_it = 0;
        for (size_t iov_it = 0; iov_it < iov_cnt - 1; ++iov_it) {
            m_iov[iov_it].buffer = (char *)(buf) + iov_length_it;
            m_iov[iov_it].length = iov_length;
            iov_length_it += iov_length;
        }

        /* Last entry */
        m_iov[iov_cnt - 1].buffer = (char *)(buf) + iov_length_it;
        m_iov[iov_cnt - 1].length = length - iov_length_it;

        m_buf   = m_iov;
        m_count = iov_cnt;
        break;
    }
    default:
        m_buf   = NULL;
        m_count = 0;
        EXPECT_TRUE(false) << "Unsupported datatype";
        break;
    }

    return *this;
}

} // ucp
