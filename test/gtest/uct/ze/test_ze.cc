/**
* Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct/test_p2p_rma.h"

#include <common/mem_buffer.h>

extern "C" {
#include <uct/ze/base/ze_base.h>
#include <uct/ze/copy/ze_copy_iface.h>
#include <uct/ze/copy/ze_copy_md.h>
}


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

    /* Device memory allocated outside UCX in its own context, as an
     * application such as a framework would allocate it */
    class external_buffer {
    public:
        external_buffer(ze_device_handle_t device, size_t length)
        {
            ze_context_desc_t ctx_desc            = {};
            ze_device_mem_alloc_desc_t alloc_desc = {};

            ctx_desc.stype   = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
            alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;

            if (zeContextCreate(uct_ze_base_get_driver(), &ctx_desc,
                                &m_context) != ZE_RESULT_SUCCESS) {
                UCS_TEST_ABORT("zeContextCreate failed");
            }

            if (zeMemAllocDevice(m_context, &alloc_desc, length, 64, device,
                                 &m_ptr) != ZE_RESULT_SUCCESS) {
                zeContextDestroy(m_context);
                UCS_TEST_ABORT("zeMemAllocDevice failed");
            }
        }

        ~external_buffer()
        {
            zeMemFree(m_context, m_ptr);
            zeContextDestroy(m_context);
        }

        void *ptr() const
        {
            return m_ptr;
        }

    private:
        ze_context_handle_t m_context;
        void                *m_ptr;
    };

    uct_ze_copy_md_t *ze_md()
    {
        return ucs_derived_of(sender().md(), uct_ze_copy_md_t);
    }

    uct_ze_copy_iface_t *ze_iface()
    {
        return ucs_derived_of(sender().iface(), uct_ze_copy_iface_t);
    }

    /* Return a device other than the MD device, preferring one on another
     * card over another tile of the same card, or NULL if there is none */
    ze_device_handle_t other_device()
    {
        ze_device_handle_t md_device            = ze_md()->ze_device;
        const uct_ze_subdevice_t *md_subdevice  = NULL;
        ze_device_handle_t other_tile           = NULL;
        const uct_ze_subdevice_t *subdevice;
        ze_device_handle_t device;
        int id;

        for (id = 0;
             (subdevice = uct_ze_base_get_subdevice_by_global_id(id)) != NULL;
             ++id) {
            if (uct_ze_base_get_device_handle_from_subdevice(subdevice) ==
                md_device) {
                md_subdevice = subdevice;
            }
        }

        for (id = 0;
             (subdevice = uct_ze_base_get_subdevice_by_global_id(id)) != NULL;
             ++id) {
            device = uct_ze_base_get_device_handle_from_subdevice(subdevice);
            if ((device == NULL) || (device == md_device)) {
                continue;
            }

            if ((md_subdevice == NULL) ||
                (subdevice->device != md_subdevice->device)) {
                UCS_TEST_MESSAGE << "using a device on another card";
                return device;
            }

            other_tile = device;
        }

        if (other_tile != NULL) {
            UCS_TEST_MESSAGE << "using another tile of the MD card";
        }

        return other_tile;
    }

    /* Device which owns the allocation, as seen from the MD context */
    ze_device_handle_t mem_device(const void *address)
    {
        ze_memory_allocation_properties_t props = {};
        ze_device_handle_t device               = NULL;

        props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
        EXPECT_EQ(ZE_RESULT_SUCCESS,
                  zeMemGetAllocProperties(ze_md()->ze_context, address,
                                          &props, &device));
        return device;
    }

    unsigned get_queue(const void *src, const void *dst)
    {
        unsigned queue_index = UINT_MAX;

        EXPECT_EQ(UCS_OK, uct_ze_copy_iface_get_queue(ze_iface(), src, dst,
                                                      &queue_index));
        return queue_index;
    }

    ze_device_handle_t queue_device(unsigned queue_index)
    {
        return ucs_array_elem(&ze_iface()->queues, queue_index).device;
    }

    void zcopy(void *buffer, void *remote_addr, size_t length, bool is_put)
    {
        uct_iov_t iov;

        iov.buffer = buffer;
        iov.length = length;
        iov.memh   = UCT_MEM_HANDLE_NULL;
        iov.stride = 0;
        iov.count  = 1;

        if (is_put) {
            ASSERT_UCS_OK(uct_ep_put_zcopy(sender().ep(0), &iov, 1,
                                           (uintptr_t)remote_addr, 0, NULL));
        } else {
            ASSERT_UCS_OK(uct_ep_get_zcopy(sender().ep(0), &iov, 1,
                                           (uintptr_t)remote_addr, 0, NULL));
        }
    }
};

UCS_TEST_SKIP_COND_P(test_ze_copy_rma, queue_follows_buffer_device,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_PUT_ZCOPY))
{
    const size_t length       = UCS_MBYTE;
    ze_device_handle_t device = other_device();
    std::vector<uint8_t> host_buf(length), check_buf(length, 0);
    unsigned queue_index;
    size_t i;

    if (device == NULL) {
        UCS_TEST_SKIP_R("needs a second ZE device");
    }

    external_buffer other_buf(device, length);
    external_buffer md_buf(ze_md()->ze_device, length);

    /* The MD context must recognize memory from another context */
    ASSERT_EQ(device, mem_device(other_buf.ptr()));
    ASSERT_EQ(ze_md()->ze_device, mem_device(md_buf.ptr()));

    /* Host-to-host copies and buffers on the MD device use the default
     * queue */
    EXPECT_EQ(0u, get_queue(host_buf.data(), check_buf.data()));
    EXPECT_EQ(0u, get_queue(md_buf.ptr(), host_buf.data()));
    EXPECT_EQ(0u, get_queue(host_buf.data(), md_buf.ptr()));
    EXPECT_EQ(ze_md()->ze_device, queue_device(0));

    /* Both directions select one queue on the buffer device */
    queue_index = get_queue(other_buf.ptr(), host_buf.data());
    ASSERT_LT(queue_index, ucs_array_length(&ze_iface()->queues));
    EXPECT_NE(0u, queue_index);
    EXPECT_EQ(device, queue_device(queue_index));
    EXPECT_EQ(queue_index, get_queue(host_buf.data(), other_buf.ptr()));

    for (i = 0; i < length; ++i) {
        host_buf[i] = (uint8_t)(i * 7 + 1);
    }

    zcopy(host_buf.data(), other_buf.ptr(), length, true);
    zcopy(check_buf.data(), other_buf.ptr(), length, false);
    EXPECT_EQ(host_buf, check_buf);
}

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
