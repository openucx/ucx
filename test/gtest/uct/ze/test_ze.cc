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
            const p2p_resource *p2p_rsc =
                    dynamic_cast<const p2p_resource*>(rsc);
            if ((p2p_rsc != NULL) && p2p_rsc->loopback) {
                result.push_back(rsc);
            }
        }

        return result;
    }

protected:
    static size_t bounded_max(size_t min_length, size_t max_length)
    {
        /* Keep a compact sweep window for CI stability while covering range logic. */
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

        return ((sender().md_attr().access_mem_types & UCS_BIT(mem_type)) ||
                ((sender().md_attr().access_mem_types &
                  UCS_BIT(UCS_MEMORY_TYPE_HOST)) &&
                 (sender().md_attr().reg_mem_types & UCS_BIT(mem_type))));
    }

    void test_xfer(send_func_t send, size_t length, unsigned flags,
                   ucs_memory_type_t mem_type) override
    {
        mapped_buffer sendbuf(length, SEED1, sender(), 1, UCS_MEMORY_TYPE_HOST);
        mapped_buffer recvbuf(length, SEED2, receiver(), 3, mem_type);

        blocking_send(send, sender_ep(), sendbuf, recvbuf, true);
        check_buf(sendbuf, recvbuf, flags);
    }
};

UCS_TEST_P(test_ze_copy_rma, put_zcopy_ze_device)
{
    if (!check_caps(UCT_IFACE_FLAG_PUT_ZCOPY) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_DEVICE)) {
        UCS_TEST_SKIP_R("ze_copy PUT zcopy ZE device memory is not available");
    }

    size_t length =
            ucs_max((size_t)64,
                    ucs_max((size_t)1,
                            (size_t)sender().iface_attr().cap.put.min_zcopy));

    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy), length,
              TEST_UCT_FLAG_SEND_ZCOPY, UCS_MEMORY_TYPE_ZE_DEVICE);
}

UCS_TEST_P(test_ze_copy_rma, put_zcopy_ze_device_range)
{
    if (!check_caps(UCT_IFACE_FLAG_PUT_ZCOPY) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_DEVICE)) {
        UCS_TEST_SKIP_R("ze_copy PUT zcopy ZE device memory is not available");
    }

    size_t min_length =
            ucs_max((size_t)1,
                    (size_t)sender().iface_attr().cap.put.min_zcopy);
    size_t max_length = bounded_max(
            min_length, (size_t)sender().iface_attr().cap.put.max_zcopy);

    test_xfer_multi_mem_type(static_cast<send_func_t>(
                                     &uct_p2p_rma_test::put_zcopy),
                             min_length, max_length, TEST_UCT_FLAG_SEND_ZCOPY,
                             UCS_MEMORY_TYPE_ZE_DEVICE);
}

UCS_TEST_P(test_ze_copy_rma, put_short_ze_device)
{
    if (!check_caps(UCT_IFACE_FLAG_PUT_SHORT) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_DEVICE)) {
        UCS_TEST_SKIP_R("ze_copy PUT short ZE device memory is not available");
    }

    size_t length =
            ucs_max((size_t)1,
                    ucs_min((size_t)64,
                            (size_t)sender().iface_attr().cap.put.max_short));

    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::put_short), length,
              TEST_UCT_FLAG_SEND_ZCOPY, UCS_MEMORY_TYPE_ZE_DEVICE);
}

UCS_TEST_P(test_ze_copy_rma, put_short_ze_device_range)
{
    if (!check_caps(UCT_IFACE_FLAG_PUT_SHORT) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_DEVICE)) {
        UCS_TEST_SKIP_R("ze_copy PUT short ZE device memory is not available");
    }

    size_t min_length = 1;
    size_t max_length = bounded_max(
            min_length,
            ucs_min((size_t)256,
                    (size_t)sender().iface_attr().cap.put.max_short));

    test_xfer_multi_mem_type(static_cast<send_func_t>(
                                     &uct_p2p_rma_test::put_short),
                             min_length, max_length, TEST_UCT_FLAG_SEND_ZCOPY,
                             UCS_MEMORY_TYPE_ZE_DEVICE);
}

UCS_TEST_P(test_ze_copy_rma, put_zcopy_ze_managed)
{
    if (!check_caps(UCT_IFACE_FLAG_PUT_ZCOPY) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_MANAGED)) {
        UCS_TEST_SKIP_R("ze_copy PUT zcopy ZE managed memory is not available");
    }

    size_t length =
            ucs_max((size_t)64,
                    ucs_max((size_t)1,
                            (size_t)sender().iface_attr().cap.put.min_zcopy));

    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy), length,
              TEST_UCT_FLAG_SEND_ZCOPY, UCS_MEMORY_TYPE_ZE_MANAGED);
}

UCS_TEST_P(test_ze_copy_rma, put_zcopy_ze_managed_range)
{
    if (!check_caps(UCT_IFACE_FLAG_PUT_ZCOPY) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_MANAGED)) {
        UCS_TEST_SKIP_R("ze_copy PUT zcopy ZE managed memory is not available");
    }

    size_t min_length =
            ucs_max((size_t)1, (size_t)sender().iface_attr().cap.put.min_zcopy);
    size_t max_length = bounded_max(
            min_length, (size_t)sender().iface_attr().cap.put.max_zcopy);

    test_xfer_multi_mem_type(static_cast<send_func_t>(
                                     &uct_p2p_rma_test::put_zcopy),
                             min_length, max_length, TEST_UCT_FLAG_SEND_ZCOPY,
                             UCS_MEMORY_TYPE_ZE_MANAGED);
}

UCS_TEST_P(test_ze_copy_rma, put_short_ze_managed)
{
    if (!check_caps(UCT_IFACE_FLAG_PUT_SHORT) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_MANAGED)) {
        UCS_TEST_SKIP_R("ze_copy PUT short ZE managed memory is not available");
    }

    size_t length =
            ucs_max((size_t)1,
                    ucs_min((size_t)64,
                            (size_t)sender().iface_attr().cap.put.max_short));

    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::put_short), length,
              TEST_UCT_FLAG_SEND_ZCOPY, UCS_MEMORY_TYPE_ZE_MANAGED);
}

UCS_TEST_P(test_ze_copy_rma, put_short_ze_managed_range)
{
    if (!check_caps(UCT_IFACE_FLAG_PUT_SHORT) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_MANAGED)) {
        UCS_TEST_SKIP_R("ze_copy PUT short ZE managed memory is not available");
    }

    size_t min_length = 1;
    size_t max_length = bounded_max(
            min_length,
            ucs_min((size_t)256,
                    (size_t)sender().iface_attr().cap.put.max_short));

    test_xfer_multi_mem_type(static_cast<send_func_t>(
                                     &uct_p2p_rma_test::put_short),
                             min_length, max_length, TEST_UCT_FLAG_SEND_ZCOPY,
                             UCS_MEMORY_TYPE_ZE_MANAGED);
}

UCS_TEST_P(test_ze_copy_rma, put_zcopy_ze_host)
{
    if (!check_caps(UCT_IFACE_FLAG_PUT_ZCOPY) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_HOST)) {
        UCS_TEST_SKIP_R("ze_copy PUT zcopy ZE host memory is not available");
    }

    size_t length =
            ucs_max((size_t)64,
                    ucs_max((size_t)1,
                            (size_t)sender().iface_attr().cap.put.min_zcopy));

    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy), length,
              TEST_UCT_FLAG_SEND_ZCOPY, UCS_MEMORY_TYPE_ZE_HOST);
}

UCS_TEST_P(test_ze_copy_rma, put_zcopy_ze_host_range)
{
    if (!check_caps(UCT_IFACE_FLAG_PUT_ZCOPY) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_HOST)) {
        UCS_TEST_SKIP_R("ze_copy PUT zcopy ZE host memory is not available");
    }

    size_t min_length =
            ucs_max((size_t)1, (size_t)sender().iface_attr().cap.put.min_zcopy);
    size_t max_length = bounded_max(
            min_length, (size_t)sender().iface_attr().cap.put.max_zcopy);

    test_xfer_multi_mem_type(static_cast<send_func_t>(
                                     &uct_p2p_rma_test::put_zcopy),
                             min_length, max_length, TEST_UCT_FLAG_SEND_ZCOPY,
                             UCS_MEMORY_TYPE_ZE_HOST);
}

UCS_TEST_P(test_ze_copy_rma, put_short_ze_host)
{
    if (!check_caps(UCT_IFACE_FLAG_PUT_SHORT) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_HOST)) {
        UCS_TEST_SKIP_R("ze_copy PUT short ZE host memory is not available");
    }

    size_t length =
            ucs_max((size_t)1,
                    ucs_min((size_t)64,
                            (size_t)sender().iface_attr().cap.put.max_short));

    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::put_short), length,
              TEST_UCT_FLAG_SEND_ZCOPY, UCS_MEMORY_TYPE_ZE_HOST);
}

UCS_TEST_P(test_ze_copy_rma, put_short_ze_host_range)
{
    if (!check_caps(UCT_IFACE_FLAG_PUT_SHORT) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_HOST)) {
        UCS_TEST_SKIP_R("ze_copy PUT short ZE host memory is not available");
    }

    size_t min_length = 1;
    size_t max_length = bounded_max(
            min_length,
            ucs_min((size_t)256,
                    (size_t)sender().iface_attr().cap.put.max_short));

    test_xfer_multi_mem_type(static_cast<send_func_t>(
                                     &uct_p2p_rma_test::put_short),
                             min_length, max_length, TEST_UCT_FLAG_SEND_ZCOPY,
                             UCS_MEMORY_TYPE_ZE_HOST);
}

UCS_TEST_P(test_ze_copy_rma, get_zcopy_ze_device)
{
    if (!check_caps(UCT_IFACE_FLAG_GET_ZCOPY) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_DEVICE)) {
        UCS_TEST_SKIP_R("ze_copy GET zcopy ZE device memory is not available");
    }

    size_t length =
            ucs_max((size_t)64,
                    ucs_max((size_t)1,
                            (size_t)sender().iface_attr().cap.get.min_zcopy));

    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy), length,
              TEST_UCT_FLAG_RECV_ZCOPY, UCS_MEMORY_TYPE_ZE_DEVICE);
}

UCS_TEST_P(test_ze_copy_rma, get_zcopy_ze_device_range)
{
    if (!check_caps(UCT_IFACE_FLAG_GET_ZCOPY) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_DEVICE)) {
        UCS_TEST_SKIP_R("ze_copy GET zcopy ZE device memory is not available");
    }

    size_t min_length =
            ucs_max((size_t)1, (size_t)sender().iface_attr().cap.get.min_zcopy);
    size_t max_length = bounded_max(
            min_length, (size_t)sender().iface_attr().cap.get.max_zcopy);

    test_xfer_multi_mem_type(static_cast<send_func_t>(
                                     &uct_p2p_rma_test::get_zcopy),
                             min_length, max_length, TEST_UCT_FLAG_RECV_ZCOPY,
                             UCS_MEMORY_TYPE_ZE_DEVICE);
}

UCS_TEST_P(test_ze_copy_rma, get_short_ze_device)
{
    if (!check_caps(UCT_IFACE_FLAG_GET_SHORT) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_DEVICE)) {
        UCS_TEST_SKIP_R("ze_copy GET short ZE device memory is not available");
    }

    size_t length =
            ucs_max((size_t)1,
                    ucs_min((size_t)64,
                            (size_t)sender().iface_attr().cap.get.max_short));

    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::get_short), length,
              TEST_UCT_FLAG_RECV_ZCOPY, UCS_MEMORY_TYPE_ZE_DEVICE);
}

UCS_TEST_P(test_ze_copy_rma, get_short_ze_device_range)
{
    if (!check_caps(UCT_IFACE_FLAG_GET_SHORT) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_DEVICE)) {
        UCS_TEST_SKIP_R("ze_copy GET short ZE device memory is not available");
    }

    size_t min_length = 1;
    size_t max_length = bounded_max(
            min_length,
            ucs_min((size_t)256,
                    (size_t)sender().iface_attr().cap.get.max_short));

    test_xfer_multi_mem_type(static_cast<send_func_t>(
                                     &uct_p2p_rma_test::get_short),
                             min_length, max_length, TEST_UCT_FLAG_RECV_ZCOPY,
                             UCS_MEMORY_TYPE_ZE_DEVICE);
}

UCS_TEST_P(test_ze_copy_rma, get_zcopy_ze_managed)
{
    if (!check_caps(UCT_IFACE_FLAG_GET_ZCOPY) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_MANAGED)) {
        UCS_TEST_SKIP_R("ze_copy GET zcopy ZE managed memory is not available");
    }

    size_t length =
            ucs_max((size_t)64,
                    ucs_max((size_t)1,
                            (size_t)sender().iface_attr().cap.get.min_zcopy));

    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy), length,
              TEST_UCT_FLAG_RECV_ZCOPY, UCS_MEMORY_TYPE_ZE_MANAGED);
}

UCS_TEST_P(test_ze_copy_rma, get_zcopy_ze_managed_range)
{
    if (!check_caps(UCT_IFACE_FLAG_GET_ZCOPY) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_MANAGED)) {
        UCS_TEST_SKIP_R("ze_copy GET zcopy ZE managed memory is not available");
    }

    size_t min_length =
            ucs_max((size_t)1, (size_t)sender().iface_attr().cap.get.min_zcopy);
    size_t max_length = bounded_max(
            min_length, (size_t)sender().iface_attr().cap.get.max_zcopy);

    test_xfer_multi_mem_type(static_cast<send_func_t>(
                                     &uct_p2p_rma_test::get_zcopy),
                             min_length, max_length, TEST_UCT_FLAG_RECV_ZCOPY,
                             UCS_MEMORY_TYPE_ZE_MANAGED);
}

UCS_TEST_P(test_ze_copy_rma, get_short_ze_managed)
{
    if (!check_caps(UCT_IFACE_FLAG_GET_SHORT) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_MANAGED)) {
        UCS_TEST_SKIP_R("ze_copy GET short ZE managed memory is not available");
    }

    size_t length =
            ucs_max((size_t)1,
                    ucs_min((size_t)64,
                            (size_t)sender().iface_attr().cap.get.max_short));

    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::get_short), length,
              TEST_UCT_FLAG_RECV_ZCOPY, UCS_MEMORY_TYPE_ZE_MANAGED);
}

UCS_TEST_P(test_ze_copy_rma, get_short_ze_managed_range)
{
    if (!check_caps(UCT_IFACE_FLAG_GET_SHORT) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_MANAGED)) {
        UCS_TEST_SKIP_R("ze_copy GET short ZE managed memory is not available");
    }

    size_t min_length = 1;
    size_t max_length = bounded_max(
            min_length,
            ucs_min((size_t)256,
                    (size_t)sender().iface_attr().cap.get.max_short));

    test_xfer_multi_mem_type(static_cast<send_func_t>(
                                     &uct_p2p_rma_test::get_short),
                             min_length, max_length, TEST_UCT_FLAG_RECV_ZCOPY,
                             UCS_MEMORY_TYPE_ZE_MANAGED);
}

UCS_TEST_P(test_ze_copy_rma, get_zcopy_ze_host)
{
    if (!check_caps(UCT_IFACE_FLAG_GET_ZCOPY) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_HOST)) {
        UCS_TEST_SKIP_R("ze_copy GET zcopy ZE host memory is not available");
    }

    size_t length =
            ucs_max((size_t)64,
                    ucs_max((size_t)1,
                            (size_t)sender().iface_attr().cap.get.min_zcopy));

    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy), length,
              TEST_UCT_FLAG_RECV_ZCOPY, UCS_MEMORY_TYPE_ZE_HOST);
}

UCS_TEST_P(test_ze_copy_rma, get_zcopy_ze_host_range)
{
    if (!check_caps(UCT_IFACE_FLAG_GET_ZCOPY) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_HOST)) {
        UCS_TEST_SKIP_R("ze_copy GET zcopy ZE host memory is not available");
    }

    size_t min_length =
            ucs_max((size_t)1, (size_t)sender().iface_attr().cap.get.min_zcopy);
    size_t max_length = bounded_max(
            min_length, (size_t)sender().iface_attr().cap.get.max_zcopy);

    test_xfer_multi_mem_type(static_cast<send_func_t>(
                                     &uct_p2p_rma_test::get_zcopy),
                             min_length, max_length, TEST_UCT_FLAG_RECV_ZCOPY,
                             UCS_MEMORY_TYPE_ZE_HOST);
}

UCS_TEST_P(test_ze_copy_rma, get_short_ze_host)
{
    if (!check_caps(UCT_IFACE_FLAG_GET_SHORT) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_HOST)) {
        UCS_TEST_SKIP_R("ze_copy GET short ZE host memory is not available");
    }

    size_t length =
            ucs_max((size_t)1,
                    ucs_min((size_t)64,
                            (size_t)sender().iface_attr().cap.get.max_short));

    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::get_short), length,
              TEST_UCT_FLAG_RECV_ZCOPY, UCS_MEMORY_TYPE_ZE_HOST);
}

UCS_TEST_P(test_ze_copy_rma, get_short_ze_host_range)
{
    if (!check_caps(UCT_IFACE_FLAG_GET_SHORT) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_HOST)) {
        UCS_TEST_SKIP_R("ze_copy GET short ZE host memory is not available");
    }

    size_t min_length = 1;
    size_t max_length = bounded_max(
            min_length,
            ucs_min((size_t)256,
                    (size_t)sender().iface_attr().cap.get.max_short));

    test_xfer_multi_mem_type(static_cast<send_func_t>(
                                     &uct_p2p_rma_test::get_short),
                             min_length, max_length, TEST_UCT_FLAG_RECV_ZCOPY,
                             UCS_MEMORY_TYPE_ZE_HOST);
}

UCS_TEST_P(test_ze_copy_rma, ze_caps_and_mem_types)
{
    if ((sender().md() == NULL) ||
        !supports_mem_type(UCS_MEMORY_TYPE_ZE_DEVICE)) {
        UCS_TEST_SKIP_R("ze_copy ZE device memory is not available");
    }

    EXPECT_TRUE(check_caps(UCT_IFACE_FLAG_GET_ZCOPY));
    EXPECT_TRUE(check_caps(UCT_IFACE_FLAG_PUT_ZCOPY));

    EXPECT_TRUE(sender().md_attr().access_mem_types &
                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE));
    EXPECT_TRUE(sender().md_attr().reg_mem_types &
                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE));

    EXPECT_TRUE(sender().md_attr().access_mem_types &
                UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST));
    EXPECT_TRUE(sender().md_attr().reg_mem_types &
                UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST));
}

_UCT_INSTANTIATE_TEST_CASE(test_ze_copy_rma, ze_copy)
