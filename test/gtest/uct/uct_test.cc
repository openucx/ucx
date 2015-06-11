/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_test.h"

#include <ucs/gtest/test_helpers.h>
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
}

uct_test::~uct_test() {
    uct_iface_config_release(m_iface_config);
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
            status = uct_pd_open(pd_resources[i].pd_name, &pd);
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
    status = uct_iface_config_modify(m_iface_config, name.c_str(), value.c_str());

    if (status == UCS_ERR_INVALID_PARAM) {
        test_base::modify_config(name, value);
    } else if (status != UCS_OK) {
        UCS_TEST_ABORT("Couldn't modify config parameter: "
                        << name.c_str() << " to " << value.c_str());
    }
}

uct_test::entity* uct_test::create_entity(size_t rx_headroom) {
    entity *new_ent = new entity(*GetParam(), m_iface_config, rx_headroom);
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

uct_test::entity::entity(const resource& resource, uct_iface_config_t *iface_config,
                         size_t rx_headroom) {
    UCS_TEST_CREATE_HANDLE(uct_worker_h, m_worker, uct_worker_destroy,
                           uct_worker_create, NULL, UCS_THREAD_MODE_MULTI /* TODO */);

    UCS_TEST_CREATE_HANDLE(uct_pd_h, m_pd, uct_pd_close,
                           uct_pd_open, resource.pd_name.c_str());

    UCS_TEST_CREATE_HANDLE(uct_iface_h, m_iface, uct_iface_close,
                           uct_iface_open, m_pd, m_worker, resource.tl_name.c_str(),
                           resource.dev_name.c_str(), rx_headroom, iface_config);

    ucs_status_t status = uct_iface_query(m_iface, &m_iface_attr);
    ASSERT_UCS_OK(status);
}


void uct_test::entity::mem_alloc(void **address_p, size_t *length_p,
                                 size_t alignement, uct_mem_h *memh_p,
                                 uct_rkey_bundle *rkey_bundle) const {
    ucs_status_t status;
    void *rkey_buffer;
    uct_pd_attr_t pd_attr;

    status = uct_pd_query(m_pd, &pd_attr);
    ASSERT_UCS_OK(status);

    status = uct_pd_mem_alloc(m_pd, UCT_ALLOC_METHOD_DEFAULT, length_p,
                              alignement, address_p, memh_p, "test");
    ASSERT_UCS_OK(status);

    rkey_buffer = malloc(pd_attr.rkey_packed_size);

    status = uct_pd_rkey_pack(m_pd, *memh_p, rkey_buffer);
    ASSERT_UCS_OK(status);

    status = uct_pd_rkey_unpack(m_pd, rkey_buffer, rkey_bundle);
    ASSERT_UCS_OK(status);

    free(rkey_buffer);
}

void uct_test::entity::mem_free(void *address, uct_mem_h memh,
                                const uct_rkey_bundle_t& rkey) const {
    ucs_status_t status;
    uct_pd_rkey_release(m_pd, &rkey);
    status = uct_pd_mem_free(m_pd, address, memh);
    ASSERT_UCS_OK(status);
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
    struct sockaddr *addr;
    ucs_status_t status;

    status = uct_iface_query(to->iface, &iface_attr);
    ASSERT_UCS_OK(status);

    addr = (struct sockaddr*)malloc(iface_attr.ep_addr_len);

    status = uct_ep_get_address(to, addr);
    ASSERT_UCS_OK(status);

    status = uct_ep_connect_to_ep(from, addr);
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
    struct sockaddr *addr = NULL;
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

        addr = (struct sockaddr*)malloc(iface_attr().iface_addr_len);

        status = uct_iface_get_address(other.iface(), addr);
        ASSERT_UCS_OK(status);

        status = uct_ep_create_connected(iface(), addr, &ep);
        ASSERT_UCS_OK(status);
    }

    if (ep != NULL) {
        m_eps[index].reset(ep, uct_ep_destroy);
    }
    free(addr);
}

void uct_test::entity::connect_to_iface(unsigned index, entity& other) {
    ucs_status_t status;
    struct sockaddr *addr = NULL;
    uct_ep_h ep = NULL;

    reserve_ep(index);
    if (m_eps[index]) {
        return; /* Already connected */
    }

    addr = (struct sockaddr*)malloc(iface_attr().iface_addr_len);

    status = uct_iface_get_address(other.iface(), addr);
    ASSERT_UCS_OK(status);

    status = uct_ep_create_connected(iface(), addr, &ep);
    ASSERT_UCS_OK(status);

    m_eps[index].reset(ep, uct_ep_destroy);
    free(addr);
}

void uct_test::entity::flush() const {
    ucs_status_t status;
    do {
        uct_worker_progress(m_worker);
        status = uct_iface_flush(m_iface);
    } while (status == UCS_ERR_NO_RESOURCE);
    ASSERT_UCS_OK(status);
}

std::ostream& operator<<(std::ostream& os, const uct_tl_resource_desc_t& resource) {
    return os << resource.tl_name << "/" << resource.dev_name;
}

uct_test::mapped_buffer::mapped_buffer(size_t size, size_t alignment, uint64_t seed, 
                                       const entity& entity, size_t offset) :
    m_entity(entity)
{
    if (size > 0)  {
        size_t alloc_size = size + offset;
        m_entity.mem_alloc(&m_buf_real, &alloc_size, alignment, &m_memh, &m_rkey);
        m_buf = (char*)m_buf_real + offset;
        m_end = (char*)m_buf + size;
        pattern_fill(seed);
    } else {
        m_buf       = NULL;
        m_buf_real  = NULL;
        m_end       = NULL;
        m_memh      = UCT_INVALID_MEM_HANDLE;
        m_rkey.rkey = UCT_INVALID_RKEY;
        m_rkey.type = NULL;
    }
}

uct_test::mapped_buffer::~mapped_buffer() {
    if (m_memh != UCT_INVALID_MEM_HANDLE) {
        m_entity.mem_free(m_buf_real, m_memh, m_rkey);
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

void uct_test::mapped_buffer::pattern_check(void *buffer, size_t length, uint64_t seed) {
    char* end = (char*)buffer + length;
    uint64_t *ptr = (uint64_t*)buffer;
    while ((char*)(ptr + 1) <= end) {
       if (*ptr != seed) {
            UCS_TEST_ABORT("At offset " << ((char*)ptr - (char*)buffer) << ": " <<
                           "Expected: 0x" << std::hex << seed << " " <<
                           "Got: 0x" << std::hex << (*ptr) << std::dec);
        }
        seed = pat(seed);
        ++ptr;
    }

    size_t remainder = (end - (char*)ptr);
    if (remainder > 0) {
        ucs_assert(remainder < sizeof(*ptr));
        uint64_t mask = UCS_MASK_SAFE(remainder * 8 * sizeof(char));
        uint64_t value = 0;
        memcpy(&value, ptr, remainder);
        if (value != (seed & mask)) {
             UCS_TEST_ABORT("At offset " << ((char*)ptr - (char*)buffer) <<
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
    return m_memh;
}

uct_rkey_t uct_test::mapped_buffer::rkey() const {
    return m_rkey.rkey;
}

std::ostream& operator<<(std::ostream& os, const resource* resource) {
    return os << resource->name();
}

