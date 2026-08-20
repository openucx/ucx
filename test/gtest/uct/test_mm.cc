/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <uct/sm/mm/base/mm_md.h>
#include <uct/sm/mm/base/mm_ep.h>
#include <ucs/time/time.h>
#include <ucs/sys/sys.h>
}
#include "uct_p2p_test.h"
#include <common/test.h>
#include "uct_test.h"
#include <sys/mman.h>


class test_uct_mm : public uct_test {
public:

    struct mm_resource : public resource {
        std::string  shm_dir;

        mm_resource(const resource& res, const std::string& shm_dir = "") :
            resource(res.component, res.component_name, res.md_name,
                     res.local_cpus, res.tl_name, res.dev_name, res.dev_type,
                     res.sys_device),
            shm_dir(shm_dir)
        {
        }

        virtual std::string name() const {
            std::string name = resource::name();
            if (!shm_dir.empty()) {
                name += ",dir=" + shm_dir;
            }
            return name;
        }
    };

    typedef struct {
        unsigned length;
        /* data follows */
    } recv_desc_t;

    static std::vector<const resource*> enum_resources(const std::string& tl_name) {
        static std::vector<mm_resource> all_resources;

        if (all_resources.empty()) {
            std::vector<const resource*> r = uct_test::enum_resources("");
            for (std::vector<const resource*>::iterator iter = r.begin();
                 iter != r.end(); ++iter) {
                if ((*iter)->tl_name == "posix") {
                    enum_posix_variants(**iter, all_resources);
                } else {
                    all_resources.push_back(mm_resource(**iter));
                }
            }
        }

        return filter_resources(all_resources, resource::is_equal_tl_name,
                                tl_name);
    }

    test_uct_mm() : m_e1(NULL), m_e2(NULL) {
        if (GetParam()->tl_name == "posix") {
            set_posix_config();
        }
    }

    const mm_resource* GetParam() {
        return dynamic_cast<const mm_resource*>(uct_test::GetParam());
    }

    static void enum_posix_variants(const resource &res,
                                    std::vector<mm_resource> &variants) {
        variants.push_back(mm_resource(res, "."       ));
        variants.push_back(mm_resource(res, "/dev/shm"));
    }

    void set_posix_config() {
        set_config("POSIX_DIR=" + GetParam()->shm_dir);
    }

    virtual void init() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        check_skip_test();

        m_e2 = uct_test::create_entity(0);
        m_entities.push_back(m_e2);

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);
    }

    static ucs_status_t mm_am_handler(void *arg, void *data, size_t length,
                                      unsigned flags) {
        recv_desc_t *my_desc = (recv_desc_t *) arg;
        uint64_t *test_mm_hdr = (uint64_t *) data;
        uint64_t *actual_data = (uint64_t *) test_mm_hdr + 1;
        unsigned data_length = length - sizeof(test_mm_hdr);

        my_desc->length = data_length;
        if (*test_mm_hdr == 0xbeef) {
            memcpy(my_desc + 1, actual_data, data_length);
        }

        return UCS_OK;
    }

    bool check_md_caps(uint64_t flags) {
        FOR_EACH_ENTITY(iter) {
            if (!(ucs_test_all_flags((*iter)->md_attr().flags, flags))) {
                return false;
            }
        }
        return true;
    }

    void test_attach_ptr(void *ptr, void *attach_ptr, uint64_t magic)
    {
        *(uint64_t*)attach_ptr = 0;
        ucs_memory_cpu_store_fence();

        *(uint64_t*)ptr        = magic;
        ucs_memory_cpu_load_fence();

        /* Writing to *ptr should also update *attach_ptr */
        EXPECT_EQ(magic, *(uint64_t*)attach_ptr)
            << "ptr=" << ptr << " attach_ptr=" << attach_ptr;

        UCS_TEST_MESSAGE <<  std::hex << *(uint64_t*)attach_ptr;
   }

    uct_mm_md_t *md(entity *e) {
        return ucs_derived_of(e->md(), uct_mm_md_t);
    }

    void test_attach(void *ptr, uct_mem_h memh, size_t size)
    {
        uct_mm_seg_t *seg = (uct_mm_seg_t*)memh;
        ucs_status_t status;

        size_t iface_addr_len = uct_mm_md_mapper_call(md(m_e1), iface_addr_length);
        std::vector<uint8_t> iface_addr(iface_addr_len);

        status = uct_mm_md_mapper_call(md(m_e1), iface_addr_pack, &iface_addr[0]);
        ASSERT_UCS_OK(status);

        uct_mm_remote_seg_t rseg;
        status = uct_mm_md_mapper_call(md(m_e2), mem_attach, seg->seg_id, size,
                                       &iface_addr[0], &rseg);
        ASSERT_UCS_OK(status);

        test_attach_ptr(ptr, rseg.address, 0xdeadbeef11111);

        uct_mm_md_mapper_call(md(m_e2), mem_detach, &rseg);
    }

    void test_rkey(void *ptr, uct_mem_h memh, size_t size)
    {
        ucs_status_t status;

        std::vector<uint8_t> rkey_buffer(m_e1->md_attr().rkey_packed_size);

        status = uct_md_mkey_pack(m_e1->md(), memh, &rkey_buffer[0]);
        ASSERT_UCS_OK(status);

        uct_rkey_bundle_t rkey_ob;
        status = uct_rkey_unpack(GetParam()->component, &rkey_buffer[0], &rkey_ob);
        ASSERT_UCS_OK(status);

        /* For shared memory transports, rkey is the offset between local and
         * remote pointers.
         */
        test_attach_ptr(ptr, UCS_PTR_BYTE_OFFSET(ptr, rkey_ob.rkey),
                        0xdeadbeef22222);

        uct_rkey_release(GetParam()->component, &rkey_ob);
    }

    void test_memh(void *ptr, uct_mem_h memh, size_t size) {
        test_attach(ptr, memh, size);
        test_attach(ptr, memh, size);
        test_rkey(ptr, memh, size);
    }

protected:
    entity *m_e1, *m_e2;
};

UCS_TEST_SKIP_COND_P(test_uct_mm, open_for_posix,
                     check_caps(UCT_IFACE_FLAG_AM_SHORT | UCT_IFACE_FLAG_CB_SYNC))
{
    uint64_t send_data   = 0xdeadbeef;
    uint64_t test_mm_hdr = 0xbeef;
    recv_desc_t *recv_buffer;

    recv_buffer = (recv_desc_t *)malloc(sizeof(*recv_buffer) +
                                        sizeof(uint64_t));
    recv_buffer->length = 0; /* Initialize length to 0 */

    /* set a callback for the uct to invoke for receiving the data */
    uct_iface_set_am_handler(m_e2->iface(), 0, mm_am_handler , recv_buffer,
                             0);

    /* send the data */
    uct_ep_am_short(m_e1->ep(0), 0, test_mm_hdr, &send_data, sizeof(send_data));

    /* progress sender and receiver until the receiver gets the message */
    wait_for_flag(&recv_buffer->length);

    ASSERT_EQ(sizeof(send_data), recv_buffer->length);
    EXPECT_EQ(send_data, *(uint64_t*)(recv_buffer+1));

    free(recv_buffer);
}

UCS_TEST_SKIP_COND_P(test_uct_mm, alloc,
                     !check_md_caps(UCT_MD_FLAG_ALLOC)) {

    size_t size               = ucs_min(100000u, m_e1->md_attr().max_alloc);
    void *address             = NULL;
    uct_md_h md_ref           = m_e1->md();
    uct_alloc_method_t method = UCT_ALLOC_METHOD_MD;
    ucs_status_t status;
    uct_mem_alloc_params_t params;
    uct_allocated_memory_t mem;

    params.field_mask      = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS      |
                             UCT_MEM_ALLOC_PARAM_FIELD_ADDRESS    |
                             UCT_MEM_ALLOC_PARAM_FIELD_MEM_TYPE   |
                             UCT_MEM_ALLOC_PARAM_FIELD_MDS        |
                             UCT_MEM_ALLOC_PARAM_FIELD_NAME;
    params.flags           = UCT_MD_MEM_ACCESS_ALL;
    params.name            = "test_mm";
    params.mem_type        = UCS_MEMORY_TYPE_HOST;
    params.address         = address;
    params.mds.mds         = &md_ref;
    params.mds.count       = 1;

    status = uct_mem_alloc(size, &method, 1, &params, &mem);
    ASSERT_UCS_OK(status);

    test_memh(mem.address, mem.memh, mem.length);

    status = uct_mem_free(&mem);
    ASSERT_UCS_OK(status);
}

UCS_TEST_SKIP_COND_P(test_uct_mm, reg,
                     !check_md_caps(UCT_MD_FLAG_REG)) {

    size_t size = ucs_min(100000u, m_e1->md_attr().max_reg);
    ucs_status_t status;

    std::vector<uint8_t> buffer(size);

    uct_mem_h memh;
    status = uct_md_mem_reg(m_e1->md(), &buffer[0], size, UCT_MD_MEM_ACCESS_ALL,
                            &memh);
    ASSERT_UCS_OK(status);

    test_memh(&buffer[0], memh, size);

    status = uct_md_mem_dereg(m_e1->md(), memh);
    ASSERT_UCS_OK(status);
}

UCS_TEST_SKIP_COND_P(test_uct_mm, flush_no_peer_access,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    uint64_t send_data = 0xdeadbeef;
    ucs_status_t status;

    do {
        status = uct_ep_am_short(m_e1->ep(0), 0, 0xbeef, &send_data,
                                 sizeof(send_data));
        if (status == UCS_ERR_NO_RESOURCE) {
            progress();
        }
    } while (status == UCS_ERR_NO_RESOURCE);
    ASSERT_UCS_OK(status);

    uct_mm_ep_t *ep = ucs_derived_of(m_e1->ep(0), uct_mm_ep_t);
    size_t pgsz     = ucs_get_page_size();
    void *page      = (void *)((uintptr_t)ep->fifo_ctl & ~(pgsz - 1));

    ASSERT_EQ(0, mprotect(page, pgsz, PROT_NONE));
    status = uct_ep_flush(m_e1->ep(0), 0, NULL);
    ASSERT_EQ(0, mprotect(page, pgsz, PROT_READ | PROT_WRITE));
    EXPECT_EQ(UCS_OK, status);
}

static ucs_status_t mm_test_am_noop(void *arg, void *data, size_t length,
                                    unsigned flags)
{
    return UCS_OK;
}

static ucs_status_t mm_test_pending_cb(uct_pending_req_t *self)
{
    return UCS_OK;
}

static void mm_test_purge_cb(uct_pending_req_t *self, void *arg)
{
    ++(*(unsigned *)arg);
}

typedef struct {
    uct_pending_req_t req;
    uct_ep_h          ep;
    int               done;
} mm_flush_pending_ctx_t;

static ucs_status_t mm_test_flush_pending_cb(uct_pending_req_t *self)
{
    mm_flush_pending_ctx_t *ctx = ucs_container_of(self, mm_flush_pending_ctx_t,
                                                   req);
    ucs_status_t status         = uct_ep_flush(ctx->ep, 0, NULL);

    if (status == UCS_OK) {
        ctx->done = 1;
        return UCS_OK;
    }
    return status;
}

UCS_TEST_SKIP_COND_P(test_uct_mm, flush_no_peer_access_pending,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    uint64_t send_data = 0xdeadbeef;
    ucs_status_t status;
    uct_mm_ep_t *ep = ucs_derived_of(m_e1->ep(0), uct_mm_ep_t);
    uct_iface_set_am_handler(m_e2->iface(), 0, mm_test_am_noop, NULL, 0);

    do {
        status = uct_ep_am_short(m_e1->ep(0), 0, 0xbeef, &send_data,
                                 sizeof(send_data));
    } while (status == UCS_OK);
    ASSERT_EQ(UCS_ERR_NO_RESOURCE, status);
    ASSERT_FALSE(ucs_arbiter_group_is_empty(&ep->arb_group));

    size_t pgsz = ucs_get_page_size();
    void *page  = (void *)((uintptr_t)ep->fifo_ctl & ~(pgsz - 1));
    ASSERT_EQ(0, mprotect(page, pgsz, PROT_NONE));

    status = uct_ep_flush(m_e1->ep(0), 0, NULL);

    ASSERT_EQ(0, mprotect(page, pgsz, PROT_READ | PROT_WRITE));
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);

    while (!ucs_arbiter_group_is_empty(&ep->arb_group)) {
        progress();
    }
}

UCS_TEST_SKIP_COND_P(test_uct_mm, pending_purge_no_peer_access,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    uint64_t send_data = 0xdeadbeef;
    ucs_status_t status;
    uct_mm_ep_t *ep = ucs_derived_of(m_e1->ep(0), uct_mm_ep_t);
    uct_iface_set_am_handler(m_e2->iface(), 0, mm_test_am_noop, NULL, 0);
    uct_pending_req_t preq;
    unsigned purged = 0;

    do {
        status = uct_ep_am_short(m_e1->ep(0), 0, 0xbeef, &send_data,
                                 sizeof(send_data));
    } while (status == UCS_OK);
    ASSERT_EQ(UCS_ERR_NO_RESOURCE, status);

    preq.func = mm_test_pending_cb;
    ASSERT_UCS_OK(uct_ep_pending_add(m_e1->ep(0), &preq, 0));

    size_t pgsz = ucs_get_page_size();
    void *page  = (void *)((uintptr_t)ep->fifo_ctl & ~(pgsz - 1));
    ASSERT_EQ(0, mprotect(page, pgsz, PROT_NONE));

    uct_ep_pending_purge(m_e1->ep(0), mm_test_purge_cb, &purged);
    status = uct_ep_flush(m_e1->ep(0), UCT_FLUSH_FLAG_CANCEL, NULL);

    ASSERT_EQ(0, mprotect(page, pgsz, PROT_READ | PROT_WRITE));
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(1u, purged);
    EXPECT_TRUE(ucs_arbiter_group_is_empty(&ep->arb_group));
}

UCS_TEST_SKIP_COND_P(test_uct_mm, flush_pending_dispatch_completes,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    uint64_t send_data = 0xdeadbeef;
    ucs_status_t status;
    uct_mm_ep_t *ep = ucs_derived_of(m_e1->ep(0), uct_mm_ep_t);
    uct_iface_set_am_handler(m_e2->iface(), 0, mm_test_am_noop, NULL, 0);

    do {
        status = uct_ep_am_short(m_e1->ep(0), 0, 0xbeef, &send_data,
                                 sizeof(send_data));
    } while (status == UCS_OK);
    ASSERT_EQ(UCS_ERR_NO_RESOURCE, status);
    ASSERT_TRUE(ucs_arbiter_group_is_scheduled(&ep->arb_group));
    ASSERT_EQ(UCS_ERR_NO_RESOURCE, uct_ep_flush(m_e1->ep(0), 0, NULL));

    mm_flush_pending_ctx_t ctx;
    ctx.req.func = mm_test_flush_pending_cb;
    ctx.ep       = m_e1->ep(0);
    ctx.done     = 0;
    ASSERT_UCS_OK(uct_ep_pending_add(m_e1->ep(0), &ctx.req, 0));

    ucs_time_t deadline = ucs_get_time() + ucs_time_from_sec(10.0);
    while (!ctx.done && (ucs_get_time() < deadline)) {
        progress();
    }
    EXPECT_TRUE(ctx.done);
    EXPECT_TRUE(ucs_arbiter_group_is_empty(&ep->arb_group));

    if (!ucs_arbiter_group_is_empty(&ep->arb_group)) {
        unsigned purged = 0;
        uct_ep_pending_purge(m_e1->ep(0), mm_test_purge_cb, &purged);
    }
}

UCT_INSTANTIATE_MM_TEST_CASE(test_uct_mm)
