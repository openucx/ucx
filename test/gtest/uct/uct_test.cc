/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_test.h"

#include <common/test_helpers.h>
#include <algorithm>
#include <malloc.h>


#define FOR_EACH_ENTITY(_iter) \
    for (ucs::ptr_vector<entity>::const_iterator _iter = m_entities.begin(); \
         _iter != m_entities.end(); ++_iter) \


std::string resource::name() const {
    std::stringstream ss;
    ss << tl_name << "/" << dev_name;
    return ss.str();
}

uct_test::uct_test() {
    ucs_status_t status;
    status = uct_iface_config_read(GetParam()->tl_name.c_str(), NULL, NULL,
                                   &m_iface_config);
    ASSERT_UCS_OK(status);
    status = uct_pd_config_read(GetParam()->pd_name.c_str(), NULL, NULL,
                                &m_pd_config);
    ASSERT_UCS_OK(status);
}

uct_test::~uct_test() {
    uct_config_release(m_iface_config);
    uct_config_release(m_pd_config);
}

std::vector<const resource*> uct_test::enum_resources(const std::string& tl_name,
                                                      bool loopback) {
    static std::vector<resource> all_resources;

    if (all_resources.empty()) {
        uct_pd_resource_desc_t *pd_resources;
        unsigned num_pd_resources;
        uct_tl_resource_desc_t *tl_resources;
        unsigned num_tl_resources;
        ucs_status_t status;

        status = uct_query_pd_resources(&pd_resources, &num_pd_resources);
        ASSERT_UCS_OK(status);

        for (unsigned i = 0; i < num_pd_resources; ++i) {
            uct_pd_h pd;
            uct_pd_config_t *pd_config;
            status = uct_pd_config_read(pd_resources[i].pd_name, NULL, NULL,
                                        &pd_config);
            ASSERT_UCS_OK(status);

            status = uct_pd_open(pd_resources[i].pd_name, pd_config, &pd);
            uct_config_release(pd_config);
            ASSERT_UCS_OK(status);

            uct_pd_attr_t pd_attr;
            status = uct_pd_query(pd, &pd_attr);
            ASSERT_UCS_OK(status);

            status = uct_pd_query_tl_resources(pd, &tl_resources, &num_tl_resources);
            ASSERT_UCS_OK(status);

            for (unsigned j = 0; j < num_tl_resources; ++j) {
                resource rsc;
                rsc.pd_name    = pd_resources[i].pd_name,
                rsc.local_cpus = pd_attr.local_cpus,
                rsc.tl_name    = tl_resources[j].tl_name,
                rsc.dev_name   = tl_resources[j].dev_name;
                all_resources.push_back(rsc);
            }

            uct_release_tl_resource_list(tl_resources);
            uct_pd_close(pd);
        }

        uct_release_pd_resource_list(pd_resources);
    }

    return filter_resources(all_resources, tl_name);
}

void uct_test::init() {
}

void uct_test::cleanup() {
    m_entities.clear();
}

void uct_test::check_caps(uint64_t flags) {
    FOR_EACH_ENTITY(iter) {
        if (!ucs_test_all_flags((*iter)->iface_attr().cap.flags, flags)) {
            UCS_TEST_SKIP_R("unsupported");
        }
    }
}

void uct_test::modify_config(const std::string& name, const std::string& value) {
    ucs_status_t status;
    status = uct_config_modify(m_iface_config, name.c_str(), value.c_str());

    if (status == UCS_ERR_NO_ELEM) {
        status = uct_config_modify(m_pd_config, name.c_str(), value.c_str());
        if (status == UCS_ERR_NO_ELEM) {
            test_base::modify_config(name, value);
        } else if (status != UCS_OK) {
            UCS_TEST_ABORT("Couldn't modify pd config parameter: " << name.c_str() <<
                           " to " << value.c_str() << ": " << ucs_status_string(status));
        }

    } else if (status != UCS_OK) {
        UCS_TEST_ABORT("Couldn't modify iface config parameter: " << name.c_str() <<
                       " to " << value.c_str() << ": " << ucs_status_string(status));
    }
}

uct_test::entity* uct_test::create_entity(size_t rx_headroom) {
    entity *new_ent = new entity(*GetParam(), m_iface_config, rx_headroom,
                                 m_pd_config);
    return new_ent;
}

const uct_test::entity& uct_test::ent(unsigned index) const {
    return m_entities.at(index);
}

void uct_test::progress() const {
    FOR_EACH_ENTITY(iter) {
        (*iter)->progress();
    }
}

void uct_test::flush() const {

    bool flushed;
    do {
        flushed = true;
        FOR_EACH_ENTITY(iter) {
            (*iter)->progress();
            ucs_status_t status = uct_iface_flush((*iter)->iface());
            if ((status == UCS_ERR_NO_RESOURCE) || (status == UCS_INPROGRESS)) {
                flushed = false;
            } else {
                ASSERT_UCS_OK(status);
            }
        }
    } while (!flushed);
}

void uct_test::short_progress_loop(double delay_ms) const {
    ucs_time_t end_time = ucs_get_time() + ucs_time_from_msec(delay_ms * ucs::test_time_multiplier());
    while (ucs_get_time() < end_time) {
        progress();
    }
}

void uct_test::twait(int delta_ms) {
    ucs_time_t now, t1, t2;
    int left;

    now = ucs_get_time();
    left = delta_ms;
    do {
        t1 = ucs_get_time();
        usleep(1000 * left);
        t2 = ucs_get_time();
        left -= (int)ucs_time_to_msec(t2-t1);
    } while (now + ucs_time_from_msec(delta_ms) > ucs_get_time());
}

uct_test::entity::entity(const resource& resource, uct_iface_config_t *iface_config,
                         size_t rx_headroom, uct_pd_config_t *pd_config) {
    ucs_status_t status;

    UCS_TEST_CREATE_HANDLE(uct_worker_h, m_worker, uct_worker_destroy,
                           uct_worker_create, &m_async.m_async, UCS_THREAD_MODE_MULTI /* TODO */);

    UCS_TEST_CREATE_HANDLE(uct_pd_h, m_pd, uct_pd_close,
                           uct_pd_open, resource.pd_name.c_str(), pd_config);

    UCS_TEST_CREATE_HANDLE(uct_iface_h, m_iface, uct_iface_close,
                           uct_iface_open, m_pd, m_worker, resource.tl_name.c_str(),
                           resource.dev_name.c_str(), rx_headroom, iface_config);

    status = uct_iface_query(m_iface, &m_iface_attr);
    ASSERT_UCS_OK(status);
}


void uct_test::entity::mem_alloc(size_t length, uct_allocated_memory_t *mem,
                                 uct_rkey_bundle *rkey_bundle) const {
    ucs_status_t status;
    void *rkey_buffer;
    uct_pd_attr_t pd_attr;

    status = uct_pd_query(m_pd, &pd_attr);
    ASSERT_UCS_OK(status);

    status = uct_iface_mem_alloc(m_iface, length, "test", mem);
    ASSERT_UCS_OK(status);

    rkey_buffer = malloc(pd_attr.rkey_packed_size);

    status = uct_pd_mkey_pack(m_pd, mem->memh, rkey_buffer);
    ASSERT_UCS_OK(status);

    status = uct_rkey_unpack(rkey_buffer, rkey_bundle);
    ASSERT_UCS_OK(status);

    free(rkey_buffer);
}

void uct_test::entity::mem_free(const uct_allocated_memory_t *mem,
                                const uct_rkey_bundle_t& rkey) const {
    ucs_status_t status;

    status = uct_rkey_release(&rkey);
    ASSERT_UCS_OK(status);

    uct_iface_mem_free(mem);
}

void uct_test::entity::progress() const {
    uct_worker_progress(m_worker);
}

uct_pd_h uct_test::entity::pd() const {
    return m_pd;
}

uct_worker_h uct_test::entity::worker() const {
    return m_worker;
}

uct_iface_h uct_test::entity::iface() const {
    return m_iface;
}

const uct_iface_attr& uct_test::entity::iface_attr() const {
    return m_iface_attr;
}

uct_ep_h uct_test::entity::ep(unsigned index) const {
    return m_eps.at(index);
}

void uct_test::entity::reserve_ep(unsigned index) {
    if (index >= m_eps.size()) {
        m_eps.resize(index + 1);
    }
}

void uct_test::entity::connect_to_ep(uct_ep_h from, uct_ep_h to)
{
    uct_iface_attr_t iface_attr;
    uct_ep_addr_t *addr;
    ucs_status_t status;

    status = uct_iface_query(to->iface, &iface_attr);
    ASSERT_UCS_OK(status);

    addr = (uct_ep_addr_t*)malloc(iface_attr.ep_addr_len);

    status = uct_ep_get_address(to, addr);
    ASSERT_UCS_OK(status);

    status = uct_ep_connect_to_ep(from, NULL, addr);
    ASSERT_UCS_OK(status);

    free(addr);
}

void uct_test::entity::create_ep(unsigned index) {
    uct_ep_h ep = NULL;
    ucs_status_t status;

    reserve_ep(index);

    if (m_eps[index]) {
        UCS_TEST_ABORT("ep[" << index << "] already exists");
    }

    status = uct_ep_create(m_iface, &ep);
    ASSERT_UCS_OK(status);
    m_eps[index].reset(ep, uct_ep_destroy);
}

void uct_test::entity::destroy_ep(unsigned index) {
    if (!m_eps[index]) {
        UCS_TEST_ABORT("ep[" << index << "] does not exist");
    }

    m_eps[index].reset();
}

void uct_test::entity::connect(unsigned index, entity& other, unsigned other_index) {
    uct_iface_addr_t *addr = NULL;
    uct_ep_h ep = NULL;
    uct_ep_h remote_ep = NULL;
    ucs_status_t status;

    reserve_ep(index);
    if (m_eps[index]) {
        return; /* Already connected */
    }

    if (iface_attr().cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {

        other.reserve_ep(other_index);
        status = uct_ep_create(other.m_iface, &remote_ep);
        ASSERT_UCS_OK(status);
        other.m_eps[other_index].reset(remote_ep, uct_ep_destroy);

        if (&other == this) {
            connect_to_ep(remote_ep, remote_ep);
        } else {
            ucs_status_t status = uct_ep_create(m_iface, &ep);
            ASSERT_UCS_OK(status);

            connect_to_ep(ep, remote_ep);
            connect_to_ep(remote_ep, ep);
        }

    } else if (iface_attr().cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {

        addr = (uct_iface_addr_t*)malloc(iface_attr().iface_addr_len);

        status = uct_iface_get_address(other.iface(), addr);
        ASSERT_UCS_OK(status);

        status = uct_ep_create_connected(iface(), NULL, addr, &ep);
        ASSERT_UCS_OK(status);
    }

    if (ep != NULL) {
        m_eps[index].reset(ep, uct_ep_destroy);
    }
    free(addr);
}

void uct_test::entity::connect_to_iface(unsigned index, entity& other) {
    ucs_status_t status;
    uct_iface_addr_t *addr = NULL;
    uct_ep_h ep = NULL;

    reserve_ep(index);
    if (m_eps[index]) {
        return; /* Already connected */
    }

    addr = (uct_iface_addr_t*)malloc(iface_attr().iface_addr_len);

    status = uct_iface_get_address(other.iface(), addr);
    ASSERT_UCS_OK(status);

    status = uct_ep_create_connected(iface(), NULL, addr, &ep);
    ASSERT_UCS_OK(status);

    m_eps[index].reset(ep, uct_ep_destroy);
    free(addr);
}

void uct_test::entity::flush() const {
    ucs_status_t status;
    do {
        uct_worker_progress(m_worker);
        status = uct_iface_flush(m_iface);
    } while (status == UCS_INPROGRESS);
    ASSERT_UCS_OK(status);
}

std::ostream& operator<<(std::ostream& os, const uct_tl_resource_desc_t& resource) {
    return os << resource.tl_name << "/" << resource.dev_name;
}

uct_test::mapped_buffer::mapped_buffer(size_t size, uint64_t seed,
                                       const entity& entity, size_t offset) :
    m_entity(entity)
{
    if (size > 0)  {
        size_t alloc_size = size + offset;
        m_entity.mem_alloc(alloc_size, &m_mem, &m_rkey);
        m_buf = (char*)m_mem.address + offset;
        m_end = (char*)m_buf         + size;
        pattern_fill(seed);
    } else {
        m_mem.method  = UCT_ALLOC_METHOD_LAST;
        m_mem.address = NULL;
        m_mem.pd      = NULL;
        m_mem.memh    = UCT_INVALID_MEM_HANDLE;
        m_mem.length  = 0;
        m_buf         = NULL;
        m_end         = NULL;
        m_rkey.rkey   = UCT_INVALID_RKEY;
        m_rkey.handle = NULL;
        m_rkey.type   = NULL;
    }
}

uct_test::mapped_buffer::~mapped_buffer() {
    if (m_mem.method != UCT_ALLOC_METHOD_LAST) {
        m_entity.mem_free(&m_mem, m_rkey);
    }
}

void uct_test::mapped_buffer::pattern_fill(uint64_t seed) {
    uint64_t *ptr = (uint64_t*)m_buf;
    while ((char*)(ptr + 1) <= m_end) {
        *ptr = seed;
        seed = pat(seed);
        ++ptr;
    }
    memcpy(ptr, &seed, (char*)m_end - (char*)ptr);
}

void uct_test::mapped_buffer::pattern_check(uint64_t seed) {
    pattern_check(ptr(), length(), seed);
}

void uct_test::mapped_buffer::pattern_check(const void *buffer, size_t length) {
    if (length > sizeof(uint64_t)) {
        pattern_check(buffer, length, *(const uint64_t*)buffer);
    }
}

void uct_test::mapped_buffer::pattern_check(const void *buffer, size_t length,
                                            uint64_t seed) {
    const char* end = (const char*)buffer + length;
    const uint64_t *ptr = (const uint64_t*)buffer;

    while ((const char*)(ptr + 1) <= end) {
       if (*ptr != seed) {
            UCS_TEST_ABORT("At offset " << ((const char*)ptr - (const char*)buffer) << ": " <<
                           "Expected: 0x" << std::hex << seed << " " <<
                           "Got: 0x" << std::hex << (*ptr) << std::dec);
        }
        seed = pat(seed);
        ++ptr;
    }

    size_t remainder = (end - (const char*)ptr);
    if (remainder > 0) {
        ucs_assert(remainder < sizeof(*ptr));
        uint64_t mask = UCS_MASK_SAFE(remainder * 8 * sizeof(char));
        uint64_t value = 0;
        memcpy(&value, ptr, remainder);
        if (value != (seed & mask)) {
             UCS_TEST_ABORT("At offset " << ((const char*)ptr - (const char*)buffer) <<
                            " (remainder " << remainder << ") : " <<
                            "Expected: 0x" << std::hex << (seed & mask) << " " <<
                            "Mask: 0x" << std::hex << mask << " " <<
                            "Got: 0x" << std::hex << value << std::dec);
         }

    }
}

void *uct_test::mapped_buffer::ptr() const {
    return m_buf;
}

uintptr_t uct_test::mapped_buffer::addr() const {
    return (uintptr_t)m_buf;
}

size_t uct_test::mapped_buffer::length() const {
    return (char*)m_end - (char*)m_buf;
}

uint64_t uct_test::mapped_buffer::pat(uint64_t prev) {
    /* LFSR pattern */
    static const uint64_t polynom = 1337;
    return (prev << 1) | (__builtin_parityl(prev & polynom) & 1);
}

uct_mem_h uct_test::mapped_buffer::memh() const {
    return m_mem.memh;
}

uct_rkey_t uct_test::mapped_buffer::rkey() const {
    return m_rkey.rkey;
}

size_t uct_test::mapped_buffer::pack(void *dest, void *arg) {
    const mapped_buffer* buf = (const mapped_buffer*)arg;
    memcpy(dest, buf->ptr(), buf->length());
    return buf->length();
}

std::ostream& operator<<(std::ostream& os, const resource* resource) {
    return os << resource->name();
}

uct_test::entity::async_wrapper::async_wrapper()
{
    ucs_status_t status;

    /* Initialize context */
    status = ucs_async_context_init(&m_async, UCS_ASYNC_MODE_THREAD);
    if (UCS_OK != status) {
        fprintf(stderr, "Failed to init async context.\n");fflush(stderr);
    }
    ASSERT_UCS_OK(status);
}

uct_test::entity::async_wrapper::~async_wrapper()
{
    ucs_async_context_cleanup(&m_async);
}


