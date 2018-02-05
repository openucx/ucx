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
        /* Ports below 1024 are considered "privileged" (can be used only by
         * user root). Ports above and including 1024 can be used by anyone */
        ret = bind(sock_fd, (struct sockaddr*)&addr_in,
                   sizeof(struct sockaddr_in));
    } while (ret);

    ret = getsockname(sock_fd, (struct sockaddr*)&ret_addr, &len);
    EXPECT_EQ(ret, 0);
    EXPECT_LT(1023, ntohs(ret_addr.sin_port)) ;

    port = ret_addr.sin_port;
    close(sock_fd);
    return port;
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
    case UCP_DATATYPE_GENERIC:
        m_buf   = buf;
        m_count = length;
        break;
    default:
        m_buf   = NULL;
        m_count = 0;
        EXPECT_TRUE(false) << "Unsupported datatype";
        break;
    }

    return *this;
}

const uint32_t MAGIC    = 0xd7d7d7d7U;
int dt_gen_start_count  = 0;
int dt_gen_finish_count = 0;

static void* dt_common_start(size_t count)
{
    dt_gen_state *dt_state = new dt_gen_state;

    dt_state->count   = count;
    dt_state->started = 1;
    dt_state->magic   = MAGIC;
    dt_gen_start_count++;

    return dt_state;
}

static void* dt_common_start_pack(void *context, const void *buffer,
                                  size_t count)
{
    return dt_common_start(count);
}

static void* dt_common_start_unpack(void *context, void *buffer, size_t count)
{
    return dt_common_start(count);
}

template <typename T>
size_t dt_packed_size(void *state)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;

    return dt_state->count * sizeof(T);
}

template <typename T>
size_t dt_pack(void *state, size_t offset, void *dest, size_t max_length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    T *p = reinterpret_cast<T*> (dest);
    uint32_t count;

    EXPECT_GT(dt_gen_start_count, dt_gen_finish_count);
    EXPECT_EQ(1, dt_state->started);
    EXPECT_EQ(uint32_t(MAGIC), dt_state->magic);

    ucs_assert((offset % sizeof(T)) == 0);

    count = ucs_min(max_length / sizeof(T),
                    dt_state->count - (offset / sizeof(T)));
    for (unsigned i = 0; i < count; ++i) {
        p[i] = (offset / sizeof(T)) + i;
    }
    return count * sizeof(T);
}

template <typename T>
ucs_status_t dt_unpack(void *state, size_t offset, const void *src,
                       size_t length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    uint32_t count;

    EXPECT_GT(dt_gen_start_count, dt_gen_finish_count);
    EXPECT_EQ(1, dt_state->started);
    EXPECT_EQ(uint32_t(MAGIC), dt_state->magic);

    count = length / sizeof(T);
    for (unsigned i = 0; i < count; ++i) {
        T expected = (offset / sizeof(T)) + i;
        T actual   = ((T*)src)[i];
        if (actual != expected) {
            UCS_TEST_ABORT("Invalid data at index " << i << ". expected: " <<
                           expected << " actual: " << actual << " offset: " <<
                           offset << ".");
        }
    }
    return UCS_OK;
}

static ucs_status_t dt_err_unpack(void *state, size_t offset, const void *src,
                                  size_t length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;

    EXPECT_GT(dt_gen_start_count, dt_gen_finish_count);
    EXPECT_EQ(1, dt_state->started);
    EXPECT_EQ(uint32_t(MAGIC), dt_state->magic);

    return UCS_ERR_NO_MEMORY;
}

static void dt_common_finish(void *state)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;

    --dt_state->started;
    EXPECT_EQ(0, dt_state->started);
    dt_gen_finish_count++;
    delete dt_state;
}

ucp_generic_dt_ops test_dt_uint32_ops = {
    dt_common_start_pack,
    dt_common_start_unpack,
    dt_packed_size<uint32_t>,
    dt_pack<uint32_t>,
    dt_unpack<uint32_t>,
    dt_common_finish
};

ucp_generic_dt_ops test_dt_uint8_ops = {
    dt_common_start_pack,
    dt_common_start_unpack,
    dt_packed_size<uint8_t>,
    dt_pack<uint8_t>,
    dt_unpack<uint8_t>,
    dt_common_finish
};

ucp_generic_dt_ops test_dt_uint32_err_ops = {
    dt_common_start_pack,
    dt_common_start_unpack,
    dt_packed_size<uint32_t>,
    dt_pack<uint32_t>,
    dt_err_unpack,
    dt_common_finish
};

} // ucp
