/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_test.h"

extern "C" {
#include <ucs/stats/stats.h>
#include <ucs/sys/string.h>
}
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
    uct_md_h pd;

    status = uct_md_config_read(GetParam()->md_name.c_str(), NULL, NULL,
                                &m_md_config);
    ASSERT_UCS_OK(status);

    status = uct_md_open(GetParam()->md_name.c_str(), m_md_config, &pd);
    ASSERT_UCS_OK(status);

    status = uct_md_iface_config_read(pd, GetParam()->tl_name.c_str(), NULL,
                                      NULL, &m_iface_config);
    ASSERT_UCS_OK(status);
    uct_md_close(pd);
}

uct_test::~uct_test() {
    uct_config_release(m_iface_config);
    uct_config_release(m_md_config);
}

std::vector<const resource*> uct_test::enum_resources(const std::string& tl_name,
                                                      bool loopback) {
    static std::vector<resource> all_resources;

    if (all_resources.empty()) {
        uct_md_resource_desc_t *md_resources;
        unsigned num_md_resources;
        uct_tl_resource_desc_t *tl_resources;
        unsigned num_tl_resources;
        ucs_status_t status;

        status = uct_query_md_resources(&md_resources, &num_md_resources);
        ASSERT_UCS_OK(status);

        for (unsigned i = 0; i < num_md_resources; ++i) {
            uct_md_h pd;
            uct_md_config_t *md_config;
            status = uct_md_config_read(md_resources[i].md_name, NULL, NULL,
                                        &md_config);
            ASSERT_UCS_OK(status);

            status = uct_md_open(md_resources[i].md_name, md_config, &pd);
            uct_config_release(md_config);
            ASSERT_UCS_OK(status);

            uct_md_attr_t md_attr;
            status = uct_md_query(pd, &md_attr);
            ASSERT_UCS_OK(status);

            status = uct_md_query_tl_resources(pd, &tl_resources, &num_tl_resources);
            ASSERT_UCS_OK(status);

            for (unsigned j = 0; j < num_tl_resources; ++j) {
                resource rsc;
                rsc.md_name    = md_resources[i].md_name,
                rsc.local_cpus = md_attr.local_cpus,
                rsc.tl_name    = tl_resources[j].tl_name,
                rsc.dev_name   = tl_resources[j].dev_name;
                rsc.dev_type   = tl_resources[j].dev_type;
                all_resources.push_back(rsc);
            }

            uct_release_tl_resource_list(tl_resources);
            uct_md_close(pd);
        }

        uct_release_md_resource_list(md_resources);
    }

    return filter_resources(all_resources, tl_name);
}

void uct_test::init() {
}

void uct_test::cleanup() {
    FOR_EACH_ENTITY(iter) {
        (*iter)->destroy_eps();
    }
    m_entities.clear();
}

bool uct_test::is_caps_supported(uint64_t required_flags) {
    bool ret = true;

    FOR_EACH_ENTITY(iter) {
        ret &= (*iter)->is_caps_supported(required_flags);
    }

    return ret;
}

void uct_test::check_caps(uint64_t required_flags, uint64_t invalid_flags) {
    FOR_EACH_ENTITY(iter) {
        (*iter)->check_caps(required_flags, invalid_flags);
    }
}

void uct_test::modify_config(const std::string& name, const std::string& value,
                             bool optional) {
    ucs_status_t status;

    status = uct_config_modify(m_iface_config, name.c_str(), value.c_str());
    if (status == UCS_ERR_NO_ELEM) {
        status = uct_config_modify(m_md_config, name.c_str(), value.c_str());
        if (status == UCS_ERR_NO_ELEM) {
            test_base::modify_config(name, value, optional);
        } else if (status != UCS_OK) {
            UCS_TEST_ABORT("Couldn't modify pd config parameter: " << name.c_str() <<
                           " to " << value.c_str() << ": " << ucs_status_string(status));
        }

    } else if (status != UCS_OK) {
        UCS_TEST_ABORT("Couldn't modify iface config parameter: " << name.c_str() <<
                       " to " << value.c_str() << ": " << ucs_status_string(status));
    }
}

bool uct_test::get_config(const std::string& name, std::string& value) const
{
    ucs_status_t status;
    const size_t max = 1024;

    value.resize(max);
    status = uct_config_get(m_iface_config, name.c_str(),
                            const_cast<char *>(value.c_str()), max);

    if (status == UCS_ERR_NO_ELEM) {
        status = uct_config_get(m_md_config, name.c_str(),
                                const_cast<char *>(value.c_str()), max);
    }

    return (status == UCS_OK);
}

void uct_test::stats_activate()
{
    ucs_stats_cleanup();
    push_config();
    modify_config("STATS_DEST",    "file:/dev/null");
    modify_config("STATS_TRIGGER", "exit");
    ucs_stats_init();
    ASSERT_TRUE(ucs_stats_is_active());
}

void uct_test::stats_restore()
{
    ucs_stats_cleanup();
    pop_config();
    ucs_stats_init();
}

uct_test::entity* uct_test::create_entity(size_t rx_headroom) {
    uct_iface_params_t iface_params;

    memset(&iface_params, 0, sizeof(iface_params));
    iface_params.rx_headroom = rx_headroom;
    entity *new_ent = new entity(*GetParam(), m_iface_config, &iface_params,
                                 m_md_config);
    return new_ent;
}

uct_test::entity* uct_test::create_entity(uct_iface_params_t &params) {
    entity *new_ent = new entity(*GetParam(), m_iface_config, &params,
                                 m_md_config);
    return new_ent;
}

const uct_test::entity& uct_test::ent(unsigned index) const {
    return m_entities.at(index);
}

unsigned uct_test::progress() const {
    unsigned count = 0;
    FOR_EACH_ENTITY(iter) {
        count += (*iter)->progress();
    }
    return count;
}

void uct_test::flush(ucs_time_t deadline) const {

    bool flushed;
    do {
        flushed = true;
        FOR_EACH_ENTITY(iter) {
            (*iter)->progress();
            ucs_status_t status = uct_iface_flush((*iter)->iface(), 0, NULL);
            if ((status == UCS_ERR_NO_RESOURCE) || (status == UCS_INPROGRESS)) {
                flushed = false;
            } else {
                ASSERT_UCS_OK(status);
            }
        }
    } while (!flushed && (ucs_get_time() < deadline));

    EXPECT_TRUE(flushed) << "Timed out";
}

void uct_test::short_progress_loop(double delay_ms) const {
    ucs_time_t end_time = ucs_get_time() + ucs_time_from_msec(delay_ms * ucs::test_time_multiplier());
    while (ucs_get_time() < end_time) {
        progress();
    }
}

void uct_test::twait(int delta_ms) const {
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
                         uct_iface_params_t *params, uct_md_config_t *md_config) {

    ucs_status_t status;

    params->mode.device.tl_name    = const_cast<char*>(resource.tl_name.c_str());
    params->mode.device.dev_name   = const_cast<char*>(resource.dev_name.c_str());
    params->stats_root = ucs_stats_get_root();
    UCS_CPU_ZERO(&params->cpu_mask);

    UCS_TEST_CREATE_HANDLE(uct_worker_h, m_worker, uct_worker_destroy,
                           uct_worker_create, &m_async.m_async, UCS_THREAD_MODE_MULTI /* TODO */);

    UCS_TEST_CREATE_HANDLE(uct_md_h, m_md, uct_md_close,
                           uct_md_open, resource.md_name.c_str(), md_config);

    status = uct_md_query(m_md, &m_md_attr);
    ASSERT_UCS_OK(status);

    UCS_TEST_CREATE_HANDLE(uct_iface_h, m_iface, uct_iface_close,
                           uct_iface_open, m_md, m_worker, params, iface_config);

    status = uct_iface_query(m_iface, &m_iface_attr);
    ASSERT_UCS_OK(status);

    uct_iface_progress_enable(m_iface, UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
}


void uct_test::entity::mem_alloc(size_t length, uct_allocated_memory_t *mem,
                                 uct_rkey_bundle *rkey_bundle) const {
    static const char *alloc_name = "uct_test";
    ucs_status_t status;
    void *rkey_buffer;

    if (md_attr().cap.flags & (UCT_MD_FLAG_ALLOC|UCT_MD_FLAG_REG)) {
        status = uct_iface_mem_alloc(m_iface, length, UCT_MD_MEM_ACCESS_ALL,
                                     alloc_name, mem);
        ASSERT_UCS_OK(status);

        rkey_buffer = malloc(md_attr().rkey_packed_size);

        status = uct_md_mkey_pack(m_md, mem->memh, rkey_buffer);
        ASSERT_UCS_OK(status);

        status = uct_rkey_unpack(rkey_buffer, rkey_bundle);
        ASSERT_UCS_OK(status);

        free(rkey_buffer);
    } else {
        uct_alloc_method_t method = UCT_ALLOC_METHOD_MMAP;
        status = uct_mem_alloc(NULL, length, UCT_MD_MEM_ACCESS_ALL,
                               &method, 1, NULL, 0, alloc_name,
                               mem);
        ASSERT_UCS_OK(status);

        ucs_assert(mem->memh == UCT_MEM_HANDLE_NULL);

        rkey_bundle->rkey   = UCT_INVALID_RKEY;
        rkey_bundle->handle = NULL;
        rkey_bundle->type   = NULL;
    }
}

void uct_test::entity::mem_free(const uct_allocated_memory_t *mem,
                                const uct_rkey_bundle_t& rkey) const {

    if (rkey.rkey != UCT_INVALID_RKEY) {
        ucs_status_t status = uct_rkey_release(&rkey);
        ASSERT_UCS_OK(status);
    }

    if (mem->method != UCT_ALLOC_METHOD_LAST) {
        uct_iface_mem_free(mem);
    }
}

unsigned uct_test::entity::progress() const {
    unsigned count = uct_worker_progress(m_worker);
    m_async.check_miss();
    return count;
}

bool uct_test::entity::is_caps_supported(uint64_t required_flags) {
    uint64_t iface_flags = iface_attr().cap.flags;
    return ucs_test_all_flags(iface_flags, required_flags);
}

void uct_test::entity::check_caps(uint64_t required_flags,
                                  uint64_t invalid_flags)
{
    uint64_t iface_flags = iface_attr().cap.flags;
    if (!ucs_test_all_flags(iface_flags, required_flags)) {
        UCS_TEST_SKIP_R("unsupported");
    }
    if (iface_flags & invalid_flags) {
        UCS_TEST_SKIP_R("unsupported");
    }
}

uct_md_h uct_test::entity::md() const {
    return m_md;
}

const uct_md_attr& uct_test::entity::md_attr() const {
    return m_md_attr;
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

void uct_test::entity::connect_p2p_ep(uct_ep_h from, uct_ep_h to)
{
    uct_iface_attr_t iface_attr;
    uct_device_addr_t *dev_addr;
    uct_ep_addr_t *ep_addr;
    ucs_status_t status;

    status = uct_iface_query(to->iface, &iface_attr);
    ASSERT_UCS_OK(status);

    dev_addr = (uct_device_addr_t*)malloc(iface_attr.device_addr_len);
    ep_addr  = (uct_ep_addr_t*)malloc(iface_attr.ep_addr_len);

    status = uct_iface_get_device_address(to->iface, dev_addr);
    ASSERT_UCS_OK(status);

    status = uct_ep_get_address(to, ep_addr);
    ASSERT_UCS_OK(status);

    status = uct_ep_connect_to_ep(from, dev_addr, ep_addr);
    ASSERT_UCS_OK(status);

    free(ep_addr);
    free(dev_addr);
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

void uct_test::entity::destroy_eps() {
    for (unsigned index = 0; index < m_eps.size(); ++index) {
        if (!m_eps[index]) {
            continue;
        }
        m_eps[index].reset();
    }
}

void uct_test::entity::connect_to_ep(unsigned index, entity& other,
                                     unsigned other_index)
{
    ucs_status_t status;
    uct_ep_h ep, remote_ep;

    reserve_ep(index);
    if (m_eps[index]) {
        return; /* Already connected */
    }

    other.reserve_ep(other_index);
    status = uct_ep_create(other.m_iface, &remote_ep);
    ASSERT_UCS_OK(status);
    other.m_eps[other_index].reset(remote_ep, uct_ep_destroy);

    if (&other == this) {
        connect_p2p_ep(remote_ep, remote_ep);
    } else {
        ucs_status_t status = uct_ep_create(m_iface, &ep);
        ASSERT_UCS_OK(status);

        connect_p2p_ep(ep, remote_ep);
        connect_p2p_ep(remote_ep, ep);

        m_eps[index].reset(ep, uct_ep_destroy);
    }
}

void uct_test::entity::connect_to_iface(unsigned index, entity& other) {
    uct_device_addr_t *dev_addr;
    uct_iface_addr_t *iface_addr;
    ucs_status_t status;
    uct_ep_h ep;

    reserve_ep(index);
    if (m_eps[index]) {
        return; /* Already connected */
    }

    dev_addr   = (uct_device_addr_t*)malloc(other.iface_attr().device_addr_len);
    iface_addr = (uct_iface_addr_t*) malloc(other.iface_attr().iface_addr_len);

    status = uct_iface_get_device_address(other.iface(), dev_addr);
    ASSERT_UCS_OK(status);

    status = uct_iface_get_address(other.iface(), iface_addr);
    ASSERT_UCS_OK(status);

    status = uct_ep_create_connected(iface(), dev_addr, iface_addr, &ep);
    ASSERT_UCS_OK(status);

    m_eps[index].reset(ep, uct_ep_destroy);

    free(iface_addr);
    free(dev_addr);
}

void uct_test::entity::connect(unsigned index, entity& other,
                               unsigned other_index)
{
    if (iface_attr().cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        connect_to_ep(index, other, other_index);
    } else if (iface_attr().cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        connect_to_iface(index, other);
    } else {
        UCS_TEST_SKIP_R("cannot connect");
    }
}

void uct_test::entity::flush() const {
    ucs_status_t status;
    do {
        progress();
        status = uct_iface_flush(m_iface, 0, NULL);
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
        m_mem.md      = NULL;
        m_mem.memh    = UCT_MEM_HANDLE_NULL;
        m_mem.length  = 0;
        m_buf         = NULL;
        m_end         = NULL;
        m_rkey.rkey   = UCT_INVALID_RKEY;
        m_rkey.handle = NULL;
        m_rkey.type   = NULL;
    }
    m_iov.buffer = ptr();
    m_iov.length = length();
    m_iov.count  = 1;
    m_iov.stride = 0;
    m_iov.memh   = memh();
}

uct_test::mapped_buffer::~mapped_buffer() {
    m_entity.mem_free(&m_mem, m_rkey);
}

void uct_test::mapped_buffer::pattern_fill(uint64_t seed) {
    pattern_fill(m_buf, (char*)m_end - (char*)m_buf, seed);
}

void uct_test::mapped_buffer::pattern_fill(void *buffer, size_t length, uint64_t seed)
{
    uint64_t *ptr = (uint64_t*)buffer;
    char *end = (char *)buffer + length;

    while ((char*)(ptr + 1) <= end) {
        *ptr = seed;
        seed = pat(seed);
        ++ptr;
    }
    memcpy(ptr, &seed, end - (char*)ptr);
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

const uct_iov_t*  uct_test::mapped_buffer::iov() const {
    return &m_iov;
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

void uct_test::entity::async_wrapper::check_miss()
{
    ucs_async_check_miss(&m_async);
}
