/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <uct/uct_test.h>

#include <map>
#include <memory>
#include <vector>

extern "C" {
#include <uct/api/v2/uct_v2.h>
}


class test_uct_sgl : public uct_test {
public:
    static std::vector<const resource*>
    enum_resources(const std::string &tl_name)
    {
        static std::map<std::string, std::vector<resource>> all_resources;
        std::vector<resource> &variants = all_resources[tl_name];

        if (variants.empty()) {
            for (const resource *elem : uct_test::enum_resources(tl_name)) {
                for (sgl_op_t op : {SGL_OP_PUT, SGL_OP_GET}) {
                    resource rsc     = *elem;
                    rsc.variant      = op;
                    rsc.variant_name = op_name(op);
                    variants.push_back(rsc);
                }
            }
        }

        return filter_resources(variants, resource::is_equal_tl_name, "");
    }

protected:
    void init() override
    {
        uct_test::init();

        m_receiver = create_entity(0);
        m_entities.push_back(m_receiver);

        m_sender = create_entity(0);
        m_entities.push_back(m_sender);
    }

    const uct_iface_attr_v2_t &sgl_iface_attr()
    {
        if (m_sgl_attr.field_mask == 0) {
            m_sgl_attr.field_mask =
                    UCT_IFACE_ATTR_FIELD_CAP_FLAGS |
                    UCT_IFACE_ATTR_FIELD_MAX_PUT_SGL_ZCOPY_COUNT |
                    UCT_IFACE_ATTR_FIELD_MAX_GET_SGL_ZCOPY_COUNT;
            if (uct_iface_query_v2(m_sender->iface(), &m_sgl_attr) != UCS_OK) {
                m_sgl_attr.cap.flags = 0;
            }
        }

        return m_sgl_attr;
    }

    void test_sgl_various_counts()
    {
        static const size_t counts[] = {1, 2, 4, 10, 1024};
        size_t length;
        size_t max;

        connect_or_skip();
        length = elem_size(2 * UCS_KBYTE);
        max    = max_count();

        for (size_t count : counts) {
            UCS_TEST_MESSAGE << "count " << ucs_min(count, max) << " length "
                             << length;
            test_sgl(std::vector<size_t>(ucs_min(count, max), length));

            if (HasFailure() || (count >= max)) {
                break;
            }
        }
    }

    void test_sgl_various_lengths()
    {
        static const size_t sizes[] = {64, 256, UCS_KBYTE, 4 * UCS_KBYTE,
                                       16 * UCS_KBYTE};
        std::vector<size_t> lengths;

        connect_or_skip();

        for (size_t size : sizes) {
            lengths.push_back(elem_size(size));
        }

        test_sgl(std::vector<size_t>(lengths.begin(),
                                     lengths.begin() +
                                             ucs_min(lengths.size(),
                                                     max_count())));
    }

    void test_sgl_zero_count()
    {
        connect_or_skip();

        sgl_arrays sgl(*m_sender, *m_receiver, first_supported_mem_type(), op(),
                       std::vector<size_t>());

        EXPECT_EQ(UCS_OK, sgl_op(sgl));
        m_sender->flush();
    }

    void test_sgl_with_callback()
    {
        sgl_completion comp = {};
        size_t count;

        connect_or_skip();

        comp.uct.func   = completion_cb;
        comp.uct.count  = 1;
        comp.uct.status = UCS_OK;

        count = ucs_min(size_t(10), max_count());
        test_sgl(std::vector<size_t>(count, elem_size(UCS_KBYTE)), &comp);
    }

private:
    enum sgl_op_t {
        SGL_OP_PUT,
        SGL_OP_GET
    };

    struct sgl_arrays {
        sgl_arrays(const entity &local_ent, const entity &remote_ent,
                   ucs_memory_type_t type, sgl_op_t op,
                   const std::vector<size_t> &sizes)
        {
            size_t count = sizes.size();

            buffers.resize(count);
            lengths.resize(count);
            memhs.resize(count);
            remote_addrs.resize(count);
            rkeys.resize(count);

            for (size_t i = 0; i < count; ++i) {
                uint64_t local_seed  = (op == SGL_OP_PUT) ? (SEED1 + i) : SEED2;
                uint64_t remote_seed = (op == SGL_OP_PUT) ? SEED2 : (SEED1 + i);

                local.emplace_back(new mapped_buffer(sizes[i], local_seed,
                                                     local_ent, 0, type));
                remote.emplace_back(new mapped_buffer(sizes[i], remote_seed,
                                                      remote_ent, 0, type));
                buffers[i]      = local[i]->ptr();
                lengths[i]      = sizes[i];
                memhs[i]        = local[i]->memh();
                remote_addrs[i] = remote[i]->addr();
                rkeys[i]        = remote[i]->rkey();
            }
        }

        std::vector<std::unique_ptr<mapped_buffer>> local;
        std::vector<std::unique_ptr<mapped_buffer>> remote;
        std::vector<void*>                          buffers;
        std::vector<size_t>                         lengths;
        std::vector<uct_mem_h>                      memhs;
        std::vector<uint64_t>                       remote_addrs;
        std::vector<uct_rkey_t>                     rkeys;
    };

    struct sgl_completion {
        uct_completion_t uct;
        unsigned         done;
    };

    static const char *op_name(sgl_op_t op)
    {
        return (op == SGL_OP_PUT) ? "put" : "get";
    }

    sgl_op_t op() const
    {
        return static_cast<sgl_op_t>(GetParam()->variant);
    }

    void connect_or_skip()
    {
        uint64_t flag = (op() == SGL_OP_PUT) ?
                        UCT_IFACE_FLAG_V2_PUT_SGL_ZCOPY :
                        UCT_IFACE_FLAG_V2_GET_SGL_ZCOPY;

        if (!(sgl_iface_attr().cap.flags & flag) || (max_count() == 0)) {
            UCS_TEST_SKIP_R("sgl zcopy is not supported");
        }

        if (first_supported_mem_type() == UCS_MEMORY_TYPE_LAST) {
            UCS_TEST_SKIP_R("no registrable memory type");
        }

        m_sender->connect(0, *m_receiver, 0);
    }

    size_t max_count()
    {
        return (op() == SGL_OP_PUT) ?
               sgl_iface_attr().max_put_sgl_zcopy_count :
               sgl_iface_attr().max_get_sgl_zcopy_count;
    }

    size_t elem_size(size_t size)
    {
        const uct_iface_attr &attr = m_sender->iface_attr();
        size_t min_zcopy           = (op() == SGL_OP_PUT) ?
                                     attr.cap.put.min_zcopy :
                                     attr.cap.get.min_zcopy;
        size_t max_zcopy           = (op() == SGL_OP_PUT) ?
                                     attr.cap.put.max_zcopy :
                                     attr.cap.get.max_zcopy;

        return ucs_min(ucs_max(size, min_zcopy), max_zcopy);
    }

    ucs_memory_type_t first_supported_mem_type()
    {
        static const ucs_memory_type_t mem_types[] = {UCS_MEMORY_TYPE_HOST,
                                                      UCS_MEMORY_TYPE_CUDA,
                                                      UCS_MEMORY_TYPE_ROCM};

        for (ucs_memory_type_t type : mem_types) {
            if ((m_sender->md_attr().reg_mem_types & UCS_BIT(type)) &&
                mem_buffer::is_mem_type_supported(type)) {
                return type;
            }
        }

        return UCS_MEMORY_TYPE_LAST;
    }

    ucs_status_t sgl_op(const sgl_arrays &sgl, uct_completion_t *comp = NULL)
    {
        if (op() == SGL_OP_PUT) {
            return uct_ep_put_sgl_zcopy(m_sender->ep(0), sgl.buffers.data(),
                                        sgl.lengths.data(), sgl.memhs.data(),
                                        sgl.remote_addrs.data(),
                                        sgl.rkeys.data(), NULL, NULL,
                                        sgl.lengths.size(), comp);
        }

        return uct_ep_get_sgl_zcopy(m_sender->ep(0), sgl.buffers.data(),
                                    sgl.lengths.data(), sgl.memhs.data(),
                                    sgl.remote_addrs.data(), sgl.rkeys.data(),
                                    NULL, NULL, sgl.lengths.size(), comp);
    }

    void check_sgl(const sgl_arrays &sgl)
    {
        for (size_t i = 0; i < sgl.lengths.size(); ++i) {
            if (op() == SGL_OP_PUT) {
                sgl.remote[i]->pattern_check(SEED1 + i);
            } else {
                sgl.local[i]->pattern_check(SEED1 + i);
            }
        }
    }

    void test_sgl(const std::vector<size_t> &sizes,
                  sgl_completion *comp = NULL)
    {
        sgl_arrays sgl(*m_sender, *m_receiver, first_supported_mem_type(),
                       op(), sizes);
        ucs_status_t status = sgl_op(sgl, (comp == NULL) ? NULL : &comp->uct);

        ASSERT_UCS_OK_OR_INPROGRESS(status);

        if ((comp != NULL) && (status == UCS_INPROGRESS)) {
            wait_for_flag(&comp->done);
            EXPECT_EQ(1u, comp->done);
            EXPECT_UCS_OK(comp->uct.status);
        }

        m_sender->flush();
        check_sgl(sgl);
    }

    static void completion_cb(uct_completion_t *self)
    {
        ucs_container_of(self, sgl_completion, uct)->done = 1;
    }

    entity              *m_sender   = NULL;
    entity              *m_receiver = NULL;
    uct_iface_attr_v2_t m_sgl_attr  = {};

    static const uint64_t SEED1 = 0xABClu;
    static const uint64_t SEED2 = 0xDEFlu;
};

UCS_TEST_P(test_uct_sgl, iface_caps_v2)
{
    const uct_iface_attr_v2_t &attr = sgl_iface_attr();

    EXPECT_EQ(!!(attr.cap.flags & UCT_IFACE_FLAG_V2_PUT_SGL_ZCOPY),
              attr.max_put_sgl_zcopy_count > 0);
    EXPECT_EQ(!!(attr.cap.flags & UCT_IFACE_FLAG_V2_GET_SGL_ZCOPY),
              attr.max_get_sgl_zcopy_count > 0);
}

UCS_TEST_P(test_uct_sgl, various_counts) {
    test_sgl_various_counts();
}

UCS_TEST_P(test_uct_sgl, various_lengths) {
    test_sgl_various_lengths();
}

UCS_TEST_P(test_uct_sgl, zero_count) {
    test_sgl_zero_count();
}

UCS_TEST_P(test_uct_sgl, with_callback) {
    test_sgl_with_callback();
}

UCT_INSTANTIATE_TEST_CASE(test_uct_sgl)
