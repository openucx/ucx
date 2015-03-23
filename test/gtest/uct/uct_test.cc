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


std::vector<uct_resource_desc_t> uct_test::enum_resources(const std::string& tl_name) {
    static std::vector<uct_resource_desc_t> all_resources;

    if (all_resources.empty()) {
        uct_resource_desc_t *resources;
        unsigned num_resources;
        ucs_status_t status;
        uct_context_h ucth;

        status = uct_init(&ucth);
        ASSERT_UCS_OK(status);

        status = uct_query_resources(ucth, &resources, &num_resources);
        ASSERT_UCS_OK(status);

        std::copy(resources, resources + num_resources,
                  std::back_inserter(all_resources));

        uct_release_resource_list(resources);
        uct_cleanup(ucth);
    }

    std::vector<uct_resource_desc_t> result;
    for (std::vector<uct_resource_desc_t>::iterator iter = all_resources.begin();
                    iter != all_resources.end(); ++iter)
    {
        if (tl_name.empty() || (std::string(iter->tl_name) == tl_name)) {
            result.push_back(*iter);
        }
    }
    return result;
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

const uct_test::entity& uct_test::ent(unsigned index) const {
    return m_entities.at(index);
}

void uct_test::progress() const {
    FOR_EACH_ENTITY(iter) {
        (*iter)->progress();
    }
}

uct_test::entity::entity(const uct_resource_desc_t& resource, size_t rx_headroom) {
    ucs_status_t status;

    status = uct_init(&m_ucth);
    ASSERT_UCS_OK(status);

    status = uct_worker_create(m_ucth, UCT_THREAD_MODE_MULTI /* TODO */, &m_worker);
    ASSERT_UCS_OK(status);

    uct_iface_config_t *iface_config;
    status = uct_iface_config_read(m_ucth, resource.tl_name, NULL, NULL,
                                   &iface_config);
    ASSERT_UCS_OK(status);

    status = uct_iface_open(m_worker, resource.tl_name, resource.dev_name,
                            rx_headroom, iface_config, &m_iface);
    ASSERT_UCS_OK(status);

    status = uct_iface_query(m_iface, &m_iface_attr);
    ASSERT_UCS_OK(status);

    uct_iface_config_release(iface_config);
}

uct_test::entity::~entity() {
    std::for_each(m_eps.begin(), m_eps.end(), uct_ep_destroy);
    uct_iface_close(m_iface);
    uct_worker_destroy(m_worker);
    uct_cleanup(m_ucth);
}

void uct_test::entity::mem_alloc(void **address_p, size_t *length_p,
                                 size_t alignement, uct_mem_h *memh_p,
                                 uct_rkey_bundle *rkey_bundle) const {
    ucs_status_t status;
    void *rkey_buffer;
    uct_pd_attr_t pd_attr;

    status = uct_pd_query(m_iface->pd, &pd_attr);
    ASSERT_UCS_OK(status);

    status = uct_pd_mem_alloc(m_iface->pd, UCT_ALLOC_METHOD_DEFAULT, length_p,
                              alignement, address_p, memh_p, "test");
    ASSERT_UCS_OK(status);

    rkey_buffer = malloc(pd_attr.rkey_packed_size);

    status = uct_pd_rkey_pack(m_iface->pd, *memh_p, rkey_buffer);
    ASSERT_UCS_OK(status);

    status = uct_pd_rkey_unpack(m_iface->pd, rkey_buffer, rkey_bundle);
    ASSERT_UCS_OK(status);

    free(rkey_buffer);
}

void uct_test::entity::mem_free(void *address, uct_mem_h memh,
                                const uct_rkey_bundle_t& rkey) const {
    ucs_status_t status;
    uct_pd_rkey_release(m_iface->pd, const_cast<uct_rkey_bundle_t*>(&rkey));
    status = uct_pd_mem_free(m_iface->pd, address, memh);
    ASSERT_UCS_OK(status);
}

void uct_test::entity::progress() const {
    uct_worker_progress(m_worker);
}

uct_iface_h uct_test::entity::iface() const {
    return m_iface;
}

const uct_iface_attr& uct_test::entity::iface_attr() const {
    return m_iface_attr;
}

void uct_test::entity::add_ep() {
    uct_ep_h ep;
    ucs_status_t status = uct_ep_create(m_iface, &ep);
    ASSERT_UCS_OK(status);
    m_eps.push_back(ep);
}

uct_ep_h uct_test::entity::ep(unsigned index) const {
    return m_eps[index];
}

void uct_test::entity::connect(unsigned index, const entity& other, unsigned other_index) const {
    ucs_status_t status;

    uct_iface_attr_t iface_attr;
    status = uct_iface_query(other.m_iface, &iface_attr);
    ASSERT_UCS_OK(status);

    uct_iface_addr_t *iface_addr = (uct_iface_addr_t*)malloc(iface_attr.iface_addr_len);
    uct_ep_addr_t *ep_addr       = (uct_ep_addr_t*)malloc(iface_attr.ep_addr_len);

    status = uct_iface_get_address(other.m_iface, iface_addr);
    ASSERT_UCS_OK(status);

    status = uct_ep_get_address(other.m_eps[other_index], ep_addr);
    ASSERT_UCS_OK(status);

    status = uct_ep_connect_to_ep(m_eps[index], iface_addr, ep_addr);
    ASSERT_UCS_OK(status);

    free(ep_addr);
    free(iface_addr);
}

void uct_test::entity::flush() const {
    ucs_status_t status;
    do {
        uct_worker_progress(m_worker);
        status = uct_iface_flush(m_iface);
    } while (status == UCS_ERR_NO_RESOURCE);
    ASSERT_UCS_OK(status);
}

std::ostream& operator<<(std::ostream& os, const uct_resource_desc_t& resource) {
    return os << resource.tl_name << "/" << resource.dev_name;
}

uct_test::mapped_buffer::mapped_buffer(size_t size, size_t alignment,
                                       uint64_t seed, const entity& entity) :
    m_entity(entity)
{
    if (size > 0) {
        size_t alloc_size = size;
        m_entity.mem_alloc(&m_buf, &alloc_size, alignment, &m_memh, &m_rkey);
        m_end = (char*)m_buf + size;
    } else {
        m_buf       = NULL;
        m_end       = NULL;
        m_memh      = UCT_INVALID_MEM_HANDLE;
        m_rkey.rkey = UCT_INVALID_RKEY;
        m_rkey.type = NULL;
    }
    pattern_fill(seed);
}

uct_test::mapped_buffer::~mapped_buffer() {
    if (m_memh != UCT_INVALID_MEM_HANDLE) {
        m_entity.mem_free(m_buf, m_memh, m_rkey);
    }
}

void uct_test::mapped_buffer::pattern_fill(uint64_t seed) {
    /* We may fill a little more; buffer has room for it */
    for (uint64_t *ptr = (uint64_t*)m_buf; (char*)ptr < m_end; ++ptr)
    {
        *ptr = seed;
        seed = pat(seed);
    }
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
