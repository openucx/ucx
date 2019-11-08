/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <uct/sm/mm/base/mm_md.h>
#include <ucs/time/time.h>
}
#include "uct_p2p_test.h"
#include <common/test.h>
#include "uct_test.h"


class test_uct_mm : public uct_test {
public:

    struct mm_resource : public resource {
        std::string  shm_dir;
        bool         use_proc_link;

        mm_resource(const resource& res, const std::string& shm_dir = "",
                    bool use_proc_link = false) :
            resource(res.component, res.md_name, res.local_cpus, res.tl_name,
                     res.dev_name, res.dev_type),
            shm_dir(shm_dir), use_proc_link(use_proc_link)
        {
        }

        virtual std::string name() const {
            std::string name = resource::name();
            if (!shm_dir.empty()) {
                name += ",dir=" + shm_dir;
            }
            if (use_proc_link) {
                name += ",procfs";
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

        return filter_resources(all_resources, tl_name);
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
        variants.push_back(mm_resource(res, ".",        false));
        variants.push_back(mm_resource(res, ".",        true));
        variants.push_back(mm_resource(res, "/dev/shm", false));
        variants.push_back(mm_resource(res, "/dev/shm", true));
    }

    void set_posix_config() {
        set_config("DIR=" + GetParam()->shm_dir);
        set_config("USE_PROC_LINK=" +
                   std::string(GetParam()->use_proc_link ? "yes" : "no"));
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
            if (!(ucs_test_all_flags((*iter)->md_attr().cap.flags, flags))) {
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

    size_t size = ucs_min(100000u, m_e1->md_attr().cap.max_alloc);
    ucs_status_t status;

    void   *address     = NULL;
    size_t alloc_length = size;
    uct_mem_h memh;
    status = uct_md_mem_alloc(m_e1->md(), &alloc_length, &address,
                              UCT_MD_MEM_ACCESS_ALL, "test_mm", &memh);
    ASSERT_UCS_OK(status);

    test_memh(address, memh, size);

    status = uct_md_mem_free(m_e1->md(), memh);
    ASSERT_UCS_OK(status);
}

UCS_TEST_SKIP_COND_P(test_uct_mm, reg,
                     !check_md_caps(UCT_MD_FLAG_REG)) {

    size_t size = ucs_min(100000u, m_e1->md_attr().cap.max_reg);
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

_UCT_INSTANTIATE_TEST_CASE(test_uct_mm, posix)
_UCT_INSTANTIATE_TEST_CASE(test_uct_mm, sysv)
_UCT_INSTANTIATE_TEST_CASE(test_uct_mm, xpmem)
