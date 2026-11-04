/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_p2p_rma.h"

#include <musa_runtime_api.h>

class test_musa_copy : public uct_p2p_rma_test {
public:
    struct flush_completion {
        uct_completion_t uct;
        unsigned         calls;

        flush_completion() : calls(0)
        {
            uct.func   = complete;
            uct.count  = 1;
            uct.status = UCS_OK;
        }

        static void complete(uct_completion_t *comp)
        {
            flush_completion *self = ucs_container_of(
                    comp, flush_completion, uct);
            ++self->calls;
        }
    };

    /* Completion which records the MUSA device which is current when the
     * callback runs. */
    struct device_completion {
        uct_completion_t uct;
        unsigned         calls;
        int              device_seen;

        device_completion() : calls(0), device_seen(-1)
        {
            uct.func   = complete;
            uct.count  = 1;
            uct.status = UCS_OK;
        }

        static void complete(uct_completion_t *comp)
        {
            device_completion *self = ucs_container_of(comp, device_completion,
                                                        uct);
            int device = -1;

            if (musaGetDevice(&device) != musaSuccess) {
                device = -2;
            }
            self->device_seen = device;
            ++self->calls;
        }
    };

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
};

UCS_TEST_SKIP_COND_P(test_musa_copy, put_zcopy_is_async,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    mapped_buffer sendbuf(4096, SEED1, sender(), 0, UCS_MEMORY_TYPE_HOST);
    mapped_buffer recvbuf(4096, SEED2, receiver(), 0, UCS_MEMORY_TYPE_MUSA);
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                            sendbuf.memh(),
                            sender().iface_attr().cap.put.max_iov);

    ++comp()->count;
    ucs_status_t status = uct_ep_put_zcopy(sender_ep(), iov, iovcnt,
                                           recvbuf.addr(), recvbuf.rkey(),
                                           comp());
    ASSERT_EQ(UCS_INPROGRESS, status);

    flush();
    recvbuf.pattern_check(SEED1);
}

UCS_TEST_SKIP_COND_P(test_musa_copy, flush_completion_multiple_streams,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    mapped_buffer host_src(4096, SEED1, sender(), 0, UCS_MEMORY_TYPE_HOST);
    mapped_buffer device_dst(4096, SEED2, receiver(), 0, UCS_MEMORY_TYPE_MUSA);
    mapped_buffer device_src(4096, SEED3, sender(), 0, UCS_MEMORY_TYPE_MUSA);
    mapped_buffer host_dst(4096, SEED2, receiver(), 0, UCS_MEMORY_TYPE_HOST);
    uct_iov_t host_iov   = {host_src.ptr(), host_src.length(),
                            host_src.memh(), 0, 1};
    uct_iov_t device_iov = {device_src.ptr(), device_src.length(),
                            device_src.memh(), 0, 1};
    flush_completion completion;
    ucs::handle<entity*> flush_guard(&sender(), [](entity *e) { e->flush(); });

    ASSERT_EQ(UCS_INPROGRESS,
              uct_ep_put_zcopy(sender_ep(), &host_iov, 1, device_dst.addr(),
                               device_dst.rkey(), NULL));
    ASSERT_EQ(UCS_INPROGRESS,
              uct_ep_put_zcopy(sender_ep(), &device_iov, 1, host_dst.addr(),
                               host_dst.rkey(), NULL));

    ASSERT_EQ(UCS_INPROGRESS,
              uct_iface_flush(sender().iface(), 0, &completion.uct));
    flush();

    EXPECT_EQ(0, completion.uct.count);
    EXPECT_EQ(1u, completion.calls);
    EXPECT_EQ(UCS_OK, completion.uct.status);
    device_dst.pattern_check(SEED1);
    host_dst.pattern_check(SEED3);
}

UCS_TEST_SKIP_COND_P(test_musa_copy, close_with_pending_flush,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    mapped_buffer sendbuf(4096, SEED1, sender(), 0, UCS_MEMORY_TYPE_HOST);
    mapped_buffer recvbuf(4096, SEED2, receiver(), 0, UCS_MEMORY_TYPE_MUSA);
    uct_iov_t iov = {sendbuf.ptr(), sendbuf.length(), sendbuf.memh(), 0, 1};
    uct_iface_params_t params = sender().iface_params();
    uct_ep_params_t ep_params = {};
    uct_iface_config_t *config;
    uct_iface_h tl_iface;
    uct_ep_h tl_ep;
    flush_completion completion;

    ASSERT_UCS_OK(uct_md_iface_config_read(sender().md(), "musa_copy",
                                          NULL, NULL, &config));
    ucs::handle<void*> config_guard(config, uct_config_release);
    ASSERT_UCS_OK(uct_iface_open(sender().md(), sender().worker(), &params,
                                 config, &tl_iface));
    ucs::handle<uct_iface_h> iface(tl_iface, uct_iface_close);
    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    ep_params.iface      = iface;
    ASSERT_UCS_OK(uct_ep_create(&ep_params, &tl_ep));
    ucs::handle<uct_ep_h> ep(tl_ep, uct_ep_destroy);

    ASSERT_EQ(UCS_INPROGRESS,
              uct_ep_put_zcopy(ep, &iov, 1, recvbuf.addr(), recvbuf.rkey(),
                               NULL));
    ASSERT_EQ(UCS_INPROGRESS, uct_iface_flush(iface, 0, &completion.uct));

    /* Buffers and their MD outlive the separate iface and its pending DMA. */
    ep.reset();
    iface.reset();
    EXPECT_EQ(0u, completion.calls);
    EXPECT_EQ(1, completion.uct.count);
    recvbuf.pattern_check(SEED1);
}

/* Completions must run with the application device restored: the copy is
 * executed on device B while the application device is A, and the zcopy and
 * flush completion callbacks must observe A when they run. */
UCS_TEST_SKIP_COND_P(test_musa_copy, completions_run_on_application_device,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    int num_devices = 0;

    ASSERT_EQ(musaSuccess, musaGetDeviceCount(&num_devices));
    if (num_devices < 2) {
        UCS_TEST_SKIP_R("requires at least 2 MUSA devices");
    }

    /* Allocate the device buffer while device B is current, so that the copy
     * is executed on device B ... */
    ASSERT_EQ(musaSuccess, musaSetDevice(1));
    mapped_buffer sendbuf(4096, SEED1, sender(), 0, UCS_MEMORY_TYPE_HOST);
    mapped_buffer recvbuf(4096, SEED2, receiver(), 0, UCS_MEMORY_TYPE_MUSA);
    /* ... and restore the application device A. */
    ASSERT_EQ(musaSuccess, musaSetDevice(0));

    device_completion zcopy_completion;
    device_completion flush_completion;
    ucs::handle<entity*> flush_guard(&sender(), [](entity *e) { e->flush(); });
    uct_iov_t iov = {sendbuf.ptr(), sendbuf.length(), sendbuf.memh(), 0, 1};
    ASSERT_EQ(UCS_INPROGRESS,
              uct_ep_put_zcopy(sender_ep(), &iov, 1, recvbuf.addr(),
                               recvbuf.rkey(), &zcopy_completion.uct));

    ASSERT_EQ(UCS_INPROGRESS,
              uct_iface_flush(sender().iface(), 0, &flush_completion.uct));

    flush();

    EXPECT_EQ(1u, zcopy_completion.calls);
    EXPECT_EQ(1u, flush_completion.calls);
    EXPECT_EQ(0, zcopy_completion.device_seen);
    EXPECT_EQ(0, flush_completion.device_seen);
    recvbuf.pattern_check(SEED1);
}

_UCT_INSTANTIATE_TEST_CASE(test_musa_copy, musa_copy)

class test_musa_copy_limited_events : public test_musa_copy {
protected:
    void init() override
    {
        modify_config("MUSA_COPY_MAX_EVENTS", "1");
        test_musa_copy::init();
    }
};

UCS_TEST_SKIP_COND_P(test_musa_copy_limited_events,
                     event_pool_exhaustion_does_not_copy,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    mapped_buffer sendbuf(4096, SEED1, sender(), 0, UCS_MEMORY_TYPE_HOST);
    mapped_buffer recvbuf(4096, SEED2, receiver(), 0, UCS_MEMORY_TYPE_MUSA);
    mapped_buffer untouched(4096, SEED3, receiver(), 0, UCS_MEMORY_TYPE_MUSA);
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                            sendbuf.memh(),
                            sender().iface_attr().cap.put.max_iov);

    ++comp()->count;
    ucs_status_t status = uct_ep_put_zcopy(sender_ep(), iov, iovcnt,
                                           recvbuf.addr(), recvbuf.rkey(),
                                           comp());
    ASSERT_EQ(UCS_INPROGRESS, status);

    /* No progress between posts: the only descriptor is still outstanding. */
    status = uct_ep_put_zcopy(sender_ep(), iov, iovcnt, untouched.addr(),
                              untouched.rkey(), NULL);
    EXPECT_EQ(UCS_ERR_NO_MEMORY, status);

    flush();
    recvbuf.pattern_check(SEED1);
    untouched.pattern_check(SEED3);
}

UCS_TEST_SKIP_COND_P(test_musa_copy_limited_events,
                     failed_flush_does_not_complete,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    mapped_buffer sendbuf(4096, SEED1, sender(), 0, UCS_MEMORY_TYPE_HOST);
    mapped_buffer recvbuf(4096, SEED2, receiver(), 0, UCS_MEMORY_TYPE_MUSA);
    uct_iov_t iov = {sendbuf.ptr(), sendbuf.length(), sendbuf.memh(), 0, 1};
    flush_completion completion;
    ucs::handle<entity*> flush_guard(&sender(), [](entity *e) { e->flush(); });

    ASSERT_EQ(UCS_INPROGRESS,
              uct_ep_put_zcopy(sender_ep(), &iov, 1, recvbuf.addr(),
                               recvbuf.rkey(), NULL));
    EXPECT_EQ(UCS_ERR_NO_MEMORY,
              uct_iface_flush(sender().iface(), 0, &completion.uct));
    flush();
    EXPECT_EQ(0u, completion.calls);
    EXPECT_EQ(1, completion.uct.count);
    recvbuf.pattern_check(SEED1);
}

_UCT_INSTANTIATE_TEST_CASE(test_musa_copy_limited_events, musa_copy)
