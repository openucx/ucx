/**
* Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

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

    /* Device or shared memory allocated outside UCX in its own context, as
     * an application such as a framework allocates it. ze_copy has to resolve
     * such memory through the MD context, as the MD mem_query already does,
     * so this deliberately does not allocate from the MD context. */
    class external_buffer {
    public:
        external_buffer(ze_device_handle_t device, size_t length,
                        ze_memory_type_t type = ZE_MEMORY_TYPE_DEVICE)
        {
            ze_context_desc_t ctx_desc            = {};
            ze_device_mem_alloc_desc_t alloc_desc = {};
            ze_host_mem_alloc_desc_t host_desc    = {};
            ze_result_t ret;

            ctx_desc.stype   = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
            alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
            host_desc.stype  = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;

            if (zeContextCreate(uct_ze_base_get_driver(), &ctx_desc,
                                &m_context) != ZE_RESULT_SUCCESS) {
                UCS_TEST_ABORT("zeContextCreate failed");
            }

            if (type == ZE_MEMORY_TYPE_SHARED) {
                ret = zeMemAllocShared(m_context, &alloc_desc, &host_desc,
                                       length, 64, device, &m_ptr);
            } else {
                ret = zeMemAllocDevice(m_context, &alloc_desc, length, 64,
                                       device, &m_ptr);
            }

            if (ret != ZE_RESULT_SUCCESS) {
                zeContextDestroy(m_context);
                UCS_TEST_ABORT("zeMemAlloc failed");
            }
        }

        ~external_buffer()
        {
            zeMemFree(m_context, m_ptr);
            zeContextDestroy(m_context);
        }

        /* Non-copyable, a copy would free the memory twice */
        external_buffer(const external_buffer&) = delete;
        external_buffer &operator=(const external_buffer&) = delete;

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
        ze_device_handle_t md_device           = ze_md()->ze_device;
        const uct_ze_subdevice_t *md_subdevice = NULL;
        ze_device_handle_t other_tile          = NULL;
        int num_subdevices                     =
                uct_ze_base_get_num_subdevices();
        const uct_ze_subdevice_t *subdevice;
        ze_device_handle_t device;
        int id;

        for (id = 0; id < num_subdevices; ++id) {
            subdevice = uct_ze_base_get_subdevice_by_global_id(id);
            if (uct_ze_base_get_device_handle_from_subdevice(subdevice) ==
                md_device) {
                md_subdevice = subdevice;
            }
        }

        for (id = 0; id < num_subdevices; ++id) {
            subdevice = uct_ze_base_get_subdevice_by_global_id(id);
            device    = uct_ze_base_get_device_handle_from_subdevice(subdevice);
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

    /* Device which owns the allocation, as seen from the MD context, or NULL
     * if the context does not resolve the address */
    ze_device_handle_t mem_device(const void *address)
    {
        ze_memory_allocation_properties_t props = {};
        ze_device_handle_t device               = NULL;

        props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
        if ((zeMemGetAllocProperties(ze_md()->ze_context, address, &props,
                                     &device) != ZE_RESULT_SUCCESS) ||
            (props.type == ZE_MEMORY_TYPE_UNKNOWN)) {
            return NULL;
        }

        return device;
    }

    /* Skip the test if the MD context does not resolve the memory of another
     * context: the Intel driver does, the specification does not require it,
     * and the iface then just keeps using the MD device queue */
    void check_buffer_resolved(ze_device_handle_t device, const void *address)
    {
        ze_device_handle_t owner = mem_device(address);

        if (owner == NULL) {
            UCS_TEST_SKIP_R("the MD context does not resolve memory of "
                            "another context");
        }

        ASSERT_EQ(device, owner);
    }

    static void fill_pattern(std::vector<uint8_t> &buf, uint8_t seed)
    {
        size_t i;

        for (i = 0; i < buf.size(); ++i) {
            buf[i] = (uint8_t)(i * 7 + seed);
        }
    }

    void fill_and_copy_through(void *remote_addr, size_t length)
    {
        std::vector<uint8_t> host_buf(length), check_buf(length, 0);

        fill_pattern(host_buf, 1);
        zcopy(host_buf.data(), remote_addr, length, true);
        zcopy(check_buf.data(), remote_addr, length, false);
        EXPECT_EQ(host_buf, check_buf);
    }

    /* Move a pattern host -> src -> dst -> host, with the device-to-device
     * step as a put from src or as a get into dst, and check it arrived */
    void copy_through_devices(void *src, void *dst, size_t length,
                              bool is_put, uint8_t seed)
    {
        std::vector<uint8_t> host_buf(length), check_buf(length, 0);

        fill_pattern(host_buf, seed);
        zcopy(host_buf.data(), src, length, true);
        if (is_put) {
            zcopy(src, dst, length, true);
        } else {
            zcopy(dst, src, length, false);
        }

        zcopy(check_buf.data(), dst, length, false);
        EXPECT_EQ(host_buf, check_buf);
    }

    unsigned get_queue(const void *src, const void *dst)
    {
        return uct_ze_copy_iface_get_queue(ze_iface(), src, dst);
    }

    ze_device_handle_t queue_device(unsigned queue_index)
    {
        return ucs_array_elem(&ze_iface()->queues, queue_index).device;
    }

    uct_ze_copy_engine_t queue_engine(unsigned queue_index)
    {
        return ucs_array_elem(&ze_iface()->queues, queue_index).engine;
    }

#if HAVE_DECL_ZECOMMANDQUEUEGETORDINAL && HAVE_DECL_ZECOMMANDLISTGETORDINAL
    /* Check that the queue runs on the engine it was created for: a copy
     * group, dedicated when its device has one, or a compute group */
    void check_queue_engine(const uct_ze_copy_queue_t &queue)
    {
        const ze_command_queue_group_property_flags_t copy_only =
                ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
        const ze_command_queue_group_property_flags_t copy_compute =
                ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY |
                ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
        uint32_t num_groups   = 0;
        bool has_copy_only    = false;
        uint32_t cmdq_ordinal = UINT32_MAX;
        uint32_t cmdl_ordinal = UINT32_MAX;
        std::vector<ze_command_queue_group_properties_t> groups;
        ze_result_t ret;

        ASSERT_EQ(ZE_RESULT_SUCCESS,
                  zeDeviceGetCommandQueueGroupProperties(queue.device,
                                                         &num_groups, NULL));
        groups.resize(num_groups);
        for (auto &group : groups) {
            group       = {};
            group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        }
        ASSERT_EQ(ZE_RESULT_SUCCESS,
                  zeDeviceGetCommandQueueGroupProperties(queue.device,
                                                         &num_groups,
                                                         groups.data()));

        for (const auto &group : groups) {
            if ((group.flags & copy_compute) == copy_only) {
                has_copy_only = true;
            }
        }

        /* The loader reports the ordinal only if the driver implements the
         * query, which is newer than the rest of the API in use */
        ret = zeCommandQueueGetOrdinal(queue.cmdq, &cmdq_ordinal);
        if (ret == ZE_RESULT_SUCCESS) {
            ret = zeCommandListGetOrdinal(queue.cmdl, &cmdl_ordinal);
        }

        if ((ret == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) ||
            (ret == ZE_RESULT_ERROR_UNSUPPORTED_VERSION)) {
            UCS_TEST_SKIP_R("the Level Zero driver does not implement the "
                            "queue ordinal query");
        }

        ASSERT_EQ(ZE_RESULT_SUCCESS, ret);
        UCS_TEST_MESSAGE << "device " << queue.device << ": "
                         << ((queue.engine == UCT_ZE_COPY_ENGINE_COMPUTE) ?
                                     "compute" : "copy")
                         << " engine queue on group " << cmdq_ordinal
                         << " of " << num_groups
                         << (has_copy_only ? ", device has a copy engine" :
                                             "");

        /* The list must be submitted to the group of its queue */
        EXPECT_EQ(cmdq_ordinal, cmdl_ordinal);
        ASSERT_LT(cmdq_ordinal, num_groups);

        if (queue.engine == UCT_ZE_COPY_ENGINE_COMPUTE) {
            /* The group runs memory copies, so it has to copy as well */
            EXPECT_EQ(copy_compute, groups[cmdq_ordinal].flags & copy_compute);
            return;
        }

        EXPECT_TRUE(groups[cmdq_ordinal].flags &
                    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY);
        if (has_copy_only) {
            EXPECT_EQ(copy_only, groups[cmdq_ordinal].flags & copy_compute);
        }
    }
#else
    void check_queue_engine(const uct_ze_copy_queue_t &queue)
    {
        UCS_TEST_SKIP_R("needs Level Zero API 1.9 "
                        "(zeCommandQueueGetOrdinal)");
    }
#endif

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
    std::vector<uint8_t> host_buf(length), check_buf(length);
    unsigned queue_index, compute_index;

    if (device == NULL) {
        UCS_TEST_SKIP_R("needs a second ZE device");
    }

    external_buffer other_buf(device, length);
    external_buffer other_buf2(device, length);
    external_buffer md_buf(ze_md()->ze_device, length);

    check_buffer_resolved(ze_md()->ze_device, md_buf.ptr());
    check_buffer_resolved(device, other_buf.ptr());
    check_buffer_resolved(device, other_buf2.ptr());

    /* Host-to-host copies and buffers on the MD device use the default
     * queue */
    EXPECT_EQ(0u, get_queue(host_buf.data(), check_buf.data()));
    EXPECT_EQ(0u, get_queue(md_buf.ptr(), host_buf.data()));
    EXPECT_EQ(0u, get_queue(host_buf.data(), md_buf.ptr()));
    EXPECT_EQ(ze_md()->ze_device, queue_device(0));

    /* Both directions select one copy engine queue on the buffer device */
    queue_index = get_queue(other_buf.ptr(), host_buf.data());
    ASSERT_LT(queue_index, ucs_array_length(&ze_iface()->queues));
    EXPECT_NE(0u, queue_index);
    EXPECT_EQ(device, queue_device(queue_index));
    EXPECT_EQ(UCT_ZE_COPY_ENGINE_COPY, queue_engine(queue_index));
    EXPECT_EQ(queue_index, get_queue(host_buf.data(), other_buf.ptr()));

    /* Between two devices, the source device runs the copy */
    EXPECT_EQ(queue_index, get_queue(other_buf.ptr(), md_buf.ptr()));
    EXPECT_EQ(0u, get_queue(md_buf.ptr(), other_buf.ptr()));

    /* Within the other device, the copy runs on its compute engine */
    compute_index = get_queue(other_buf.ptr(), other_buf2.ptr());
    ASSERT_LT(compute_index, ucs_array_length(&ze_iface()->queues));
    EXPECT_NE(0u, compute_index);
    EXPECT_NE(queue_index, compute_index);
    EXPECT_EQ(device, queue_device(compute_index));
    EXPECT_EQ(UCT_ZE_COPY_ENGINE_COMPUTE, queue_engine(compute_index));

    /* Data moves through the copy queue and through the compute queue of
     * the other device */
    fill_and_copy_through(other_buf.ptr(), length);
    copy_through_devices(other_buf.ptr(), other_buf2.ptr(), length, true, 2);
    check_queue_engine(ucs_array_elem(&ze_iface()->queues, compute_index));
}

UCS_TEST_SKIP_COND_P(test_ze_copy_rma, md_queue_when_create_fails,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_PUT_ZCOPY))
{
    const size_t length       = 64 * UCS_KBYTE;
    ze_device_handle_t device = other_device();
    ze_bool_t can_access_peer = false;
    std::vector<uint8_t> host_buf(length);
    uct_ze_copy_queue_t *queue;

    if (device == NULL) {
        UCS_TEST_SKIP_R("needs a second ZE device");
    }

    /* The MD device queue can move the data only if it reaches the memory
     * of the other device */
    ASSERT_EQ(ZE_RESULT_SUCCESS,
              zeDeviceCanAccessPeer(ze_md()->ze_device, device,
                                    &can_access_peer));
    if (!can_access_peer) {
        UCS_TEST_SKIP_R("the MD device has no peer access to the other "
                        "device");
    }

    external_buffer other_buf(device, length);
    check_buffer_resolved(device, other_buf.ptr());

    /* Insert the entry which get_queue records after a failed queue
     * creation; the failure itself cannot be provoked here. Checks the use
     * of the entry and that iface cleanup skips it, not its creation. */
    queue = ucs_array_append(&ze_iface()->queues,
                             UCS_TEST_ABORT("cannot append a queue"));
    queue->device = device;
    queue->engine = UCT_ZE_COPY_ENGINE_COPY;
    queue->cmdq   = NULL;
    queue->cmdl   = NULL;

    /* Copies with the buffer of that device fall back to the MD device
     * queue, in both directions, and still move the data */
    EXPECT_EQ(0u, get_queue(other_buf.ptr(), host_buf.data()));
    EXPECT_EQ(0u, get_queue(host_buf.data(), other_buf.ptr()));
    EXPECT_EQ(ze_md()->ze_device, queue_device(0));

    fill_and_copy_through(other_buf.ptr(), length);
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

UCS_TEST_P(test_ze_copy_rma, queue_on_copy_engine)
{
    ze_device_handle_t device = other_device();
    std::vector<uint8_t> host_buf(64);
    const uct_ze_copy_queue_t *queue;
    ze_device_handle_t owner;

    /* Also create the queue of another device, if there is one and the MD
     * context resolves its memory; the queue stays in the iface after the
     * buffer is released */
    if (device != NULL) {
        external_buffer other_buf(device, host_buf.size());

        owner = mem_device(other_buf.ptr());
        if (owner == NULL) {
            UCS_TEST_MESSAGE << "the MD context does not resolve memory of "
                                "another context, checking the MD queue only";
        } else {
            ASSERT_EQ(device, owner);
            EXPECT_NE(0u, get_queue(other_buf.ptr(), host_buf.data()));
        }
    }

    ucs_array_for_each(queue, &ze_iface()->queues) {
        if (queue->cmdq == NULL) {
            /* Device whose queue could not be created, copies on it use the
             * MD device queue; see md_queue_when_create_fails */
            UCS_TEST_MESSAGE << "device " << queue->device
                             << ": no queue of its own";
            continue;
        }

        check_queue_engine(*queue);
    }
}

UCS_TEST_SKIP_COND_P(test_ze_copy_rma, same_device_copy_on_compute_engine,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_PUT_ZCOPY))
{
    const size_t length       = UCS_MBYTE;
    ze_device_handle_t device = ze_md()->ze_device;
    std::vector<uint8_t> host_buf(length);
    unsigned queue_index;

    external_buffer src_buf(device, length);
    external_buffer dst_buf(device, length);

    check_buffer_resolved(device, src_buf.ptr());
    check_buffer_resolved(device, dst_buf.ptr());

    /* Copies to and from host memory stay on the copy engine queue */
    EXPECT_EQ(0u, get_queue(src_buf.ptr(), host_buf.data()));
    EXPECT_EQ(0u, get_queue(host_buf.data(), src_buf.ptr()));
    EXPECT_EQ(UCT_ZE_COPY_ENGINE_COPY, queue_engine(0));

    /* A copy within the device selects the compute engine queue of the
     * device, created on first use, in both directions */
    queue_index = get_queue(src_buf.ptr(), dst_buf.ptr());
    ASSERT_LT(queue_index, ucs_array_length(&ze_iface()->queues));
    EXPECT_NE(0u, queue_index);
    EXPECT_EQ(device, queue_device(queue_index));
    EXPECT_EQ(UCT_ZE_COPY_ENGINE_COMPUTE, queue_engine(queue_index));
    EXPECT_EQ(queue_index, get_queue(dst_buf.ptr(), src_buf.ptr()));

    /* Move data between the two device buffers with a put and with a get */
    copy_through_devices(src_buf.ptr(), dst_buf.ptr(), length, true, 1);
    copy_through_devices(src_buf.ptr(), dst_buf.ptr(), length, false, 3);

    check_queue_engine(ucs_array_elem(&ze_iface()->queues, queue_index));
}

UCS_TEST_SKIP_COND_P(test_ze_copy_rma, shared_memory_copy_on_copy_engine,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_PUT_ZCOPY))
{
    const size_t length       = 64 * UCS_KBYTE;
    ze_device_handle_t device = ze_md()->ze_device;

    external_buffer device_buf(device, length);
    external_buffer shared_buf(device, length, ZE_MEMORY_TYPE_SHARED);

    check_buffer_resolved(device, device_buf.ptr());
    check_buffer_resolved(device, shared_buf.ptr());

    /* Shared memory may reside on the host, so a copy with it is not a copy
     * within the device and stays on the copy engine queue */
    EXPECT_EQ(0u, get_queue(device_buf.ptr(), shared_buf.ptr()));
    EXPECT_EQ(0u, get_queue(shared_buf.ptr(), device_buf.ptr()));

    fill_and_copy_through(shared_buf.ptr(), length);
}

UCT_INSTANTIATE_ZE_TEST_CASE(test_ze_copy_rma)
