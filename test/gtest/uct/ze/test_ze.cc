/**
* Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct/test_p2p_rma.h"

#include <common/mem_buffer.h>


class test_ze_copy_rma : public uct_p2p_rma_test {
public:
    static std::vector<const resource*>
    enum_resources(const std::string &tl_name)
    {
        std::vector<const resource*> resources =
                uct_p2p_rma_test::enum_resources(tl_name);
        std::vector<const resource*> result;

        for (const resource *rsc : resources) {
            const p2p_resource *p2p_rsc = dynamic_cast<const p2p_resource*>(
                    rsc);
            if ((p2p_rsc != NULL) && p2p_rsc->loopback) {
                result.push_back(rsc);
            }
        }

        return result;
    }

protected:
    static const std::vector<ucs_memory_type_t> &ze_mem_types()
    {
        static const std::vector<ucs_memory_type_t> types =
                {UCS_MEMORY_TYPE_ZE_HOST, UCS_MEMORY_TYPE_ZE_DEVICE,
                 UCS_MEMORY_TYPE_ZE_MANAGED};

        return types;
    }

    static size_t bounded_max(size_t min_length, size_t max_length)
    {
        /* Compact sweep window for CI stability while covering range logic. */
        return ucs_min(max_length, min_length + 1024);
    }

    void init() override
    {
        uct_p2p_rma_test::init();

        if (sender().md() == NULL) {
            UCS_TEST_SKIP_R("ze_copy MD is not available");
        }
    }

    bool supports_mem_type(ucs_memory_type_t mem_type)
    {
        if (sender().md() == NULL) {
            return false;
        }

        if (!mem_buffer::is_mem_type_supported(mem_type)) {
            return false;
        }

        return ((sender().md_attr().access_mem_types & UCS_BIT(mem_type)) ||
                ((sender().md_attr().access_mem_types &
                  UCS_BIT(UCS_MEMORY_TYPE_HOST)) &&
                 (sender().md_attr().reg_mem_types & UCS_BIT(mem_type))));
    }

    /* Run a single-length transfer for every supported ZE memory type */
    void run_single(send_func_t send, unsigned flags, size_t length)
    {
        size_t tested = 0;

        for (auto mem_type : ze_mem_types()) {
            if (!supports_mem_type(mem_type)) {
                UCS_TEST_MESSAGE << "skipping "
                                 << ucs_memory_type_names[mem_type]
                                 << " (unsupported by system or MD)";
                continue;
            }

            test_xfer(send, length, flags, mem_type);
            ++tested;
        }

        if (tested == 0) {
            UCS_TEST_SKIP_R("No supported ZE memory types");
        }
    }

    /* Run a length-range transfer for every supported ZE memory type */
    void run_range(send_func_t send, unsigned flags, size_t min_length,
                   size_t max_length)
    {
        size_t tested = 0;

        for (auto mem_type : ze_mem_types()) {
            if (!supports_mem_type(mem_type)) {
                UCS_TEST_MESSAGE << "skipping "
                                 << ucs_memory_type_names[mem_type]
                                 << " (unsupported by system or MD)";
                continue;
            }

            test_xfer_multi_mem_type(send, min_length, max_length, flags,
                                     mem_type);
            ++tested;
        }

        if (tested == 0) {
            UCS_TEST_SKIP_R("No supported ZE memory types");
        }
    }
};

UCS_TEST_SKIP_COND_P(test_ze_copy_rma, put_zcopy,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    size_t min_zcopy = sender().iface_attr().cap.put.min_zcopy;
    size_t length    = ucs_max(64ul, ucs_max(1ul, min_zcopy));

    run_single(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy),
               TEST_UCT_FLAG_SEND_ZCOPY, length);
}

UCS_TEST_SKIP_COND_P(test_ze_copy_rma, put_zcopy_range,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    const uct_iface_attr_t &attr = sender().iface_attr();
    size_t max_zcopy             = attr.cap.put.max_zcopy;
    size_t min_length            = ucs_max(1ul, attr.cap.put.min_zcopy);
    size_t max_length            = bounded_max(min_length, max_zcopy);

    run_range(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy),
              TEST_UCT_FLAG_SEND_ZCOPY, min_length, max_length);
}

UCS_TEST_SKIP_COND_P(test_ze_copy_rma, put_short,
                     !check_caps(UCT_IFACE_FLAG_PUT_SHORT))
{
    size_t max_short = sender().iface_attr().cap.put.max_short;
    size_t length    = ucs_max(1ul, ucs_min(64ul, max_short));

    run_single(static_cast<send_func_t>(&uct_p2p_rma_test::put_short),
               TEST_UCT_FLAG_SEND_ZCOPY, length);
}

UCS_TEST_SKIP_COND_P(test_ze_copy_rma, put_short_range,
                     !check_caps(UCT_IFACE_FLAG_PUT_SHORT))
{
    const uct_iface_attr_t &attr = sender().iface_attr();
    size_t max_short             = ucs_min(256ul, attr.cap.put.max_short);
    size_t min_length            = 1;
    size_t max_length            = bounded_max(min_length, max_short);

    run_range(static_cast<send_func_t>(&uct_p2p_rma_test::put_short),
              TEST_UCT_FLAG_SEND_ZCOPY, min_length, max_length);
}

UCS_TEST_SKIP_COND_P(test_ze_copy_rma, get_zcopy,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY))
{
    size_t min_zcopy = sender().iface_attr().cap.get.min_zcopy;
    size_t length    = ucs_max(64ul, ucs_max(1ul, min_zcopy));

    run_single(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
               TEST_UCT_FLAG_RECV_ZCOPY, length);
}

UCS_TEST_SKIP_COND_P(test_ze_copy_rma, get_zcopy_range,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY))
{
    const uct_iface_attr_t &attr = sender().iface_attr();
    size_t max_zcopy             = attr.cap.get.max_zcopy;
    size_t min_length            = ucs_max(1ul, attr.cap.get.min_zcopy);
    size_t max_length            = bounded_max(min_length, max_zcopy);

    run_range(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
              TEST_UCT_FLAG_RECV_ZCOPY, min_length, max_length);
}

UCS_TEST_SKIP_COND_P(test_ze_copy_rma, get_short,
                     !check_caps(UCT_IFACE_FLAG_GET_SHORT))
{
    size_t max_short = sender().iface_attr().cap.get.max_short;
    size_t length    = ucs_max(1ul, ucs_min(64ul, max_short));

    run_single(static_cast<send_func_t>(&uct_p2p_rma_test::get_short),
               TEST_UCT_FLAG_RECV_ZCOPY, length);
}

UCS_TEST_SKIP_COND_P(test_ze_copy_rma, get_short_range,
                     !check_caps(UCT_IFACE_FLAG_GET_SHORT))
{
    const uct_iface_attr_t &attr = sender().iface_attr();
    size_t max_short             = ucs_min(256ul, attr.cap.get.max_short);
    size_t min_length            = 1;
    size_t max_length            = bounded_max(min_length, max_short);

    run_range(static_cast<send_func_t>(&uct_p2p_rma_test::get_short),
              TEST_UCT_FLAG_RECV_ZCOPY, min_length, max_length);
}

UCS_TEST_SKIP_COND_P(test_ze_copy_rma, ze_caps_and_mem_types,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_PUT_ZCOPY))
{
    EXPECT_TRUE(sender().md_attr().access_mem_types &
                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE));
    EXPECT_TRUE(sender().md_attr().reg_mem_types &
                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE));

    EXPECT_TRUE(sender().md_attr().access_mem_types &
                UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST));
    EXPECT_TRUE(sender().md_attr().reg_mem_types &
                UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST));
}

UCT_INSTANTIATE_ZE_TEST_CASE(test_ze_copy_rma)
