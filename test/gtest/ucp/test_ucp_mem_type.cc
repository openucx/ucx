/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
* Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"
#include <common/mem_buffer.h>

#include <memory>

extern "C" {
#include <uct/api/uct.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_worker.inl>
#include <ucp/proto/proto.h>
#include <ucp/proto/proto_select.inl>
#include <ucp/core/ucp_ep.inl>
#include <ucp/dt/dt.h>
}


class test_ucp_mem_type : public ucp_test {
public:
    static void get_test_variants_base(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_TAG);
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_memtypes(variants, get_test_variants_base);
    }

protected:
    ucs_memory_type_t mem_type() const {
        return static_cast<ucs_memory_type_t>(get_variant_value());
    }
};

UCS_TEST_P(test_ucp_mem_type, detect) {

    const size_t size                      = 256;
    const ucs_memory_type_t alloc_mem_type = mem_type();
    ucp_memory_info_t mem_info;

    mem_buffer b(size, alloc_mem_type);

    ucp_memory_detect(sender().ucph(), b.ptr(), size, &mem_info);
    EXPECT_EQ(alloc_mem_type, mem_info.type);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type, all, "all")

class test_ucp_mem_type_alloc_before_init : public test_ucp_mem_type {
public:
    test_ucp_mem_type_alloc_before_init() {
        m_size = 10000;
    }

    virtual void init() {
        m_send_buffer.reset(new mem_buffer(m_size, mem_type()));
        m_recv_buffer.reset(new mem_buffer(m_size, mem_type()));
        test_ucp_mem_type::init();
    }

    virtual void cleanup() {
        test_ucp_mem_type::cleanup();
        m_send_buffer.reset();
        m_recv_buffer.reset();
    }

    static const uint64_t SEED = 0x1111111111111111lu;
protected:
    size_t                     m_size;
    ucs::auto_ptr<mem_buffer>  m_send_buffer, m_recv_buffer;
};

UCS_TEST_P(test_ucp_mem_type_alloc_before_init, xfer) {
    sender().connect(&receiver(), get_ep_params());

    ucp_memory_info_t mem_info;
    ucp_memory_detect(sender().ucph(), m_send_buffer->ptr(), m_size, &mem_info);
    EXPECT_EQ(mem_type(), mem_info.type) << "send buffer";
    ucp_memory_detect(receiver().ucph(), m_recv_buffer->ptr(), m_size,
                      &mem_info);
    EXPECT_EQ(mem_type(), mem_info.type) << "receive buffer";

    mem_buffer::pattern_fill(m_send_buffer->ptr(), m_size, SEED, mem_type());

    for (int i = 0; i < 3; ++i) {
        mem_buffer::pattern_fill(m_recv_buffer->ptr(), m_size, 0, mem_type());

        void *sreq = ucp_tag_send_nb(sender().ep(), m_send_buffer->ptr(), m_size,
                                     ucp_dt_make_contig(1), 1,
                                     (ucp_send_callback_t)ucs_empty_function);
        void *rreq = ucp_tag_recv_nb(receiver().worker(), m_recv_buffer->ptr(),
                                     m_size, ucp_dt_make_contig(1), 1, 1,
                                     (ucp_tag_recv_callback_t)ucs_empty_function);
        request_wait(sreq);
        request_wait(rreq);

        mem_buffer::pattern_check(m_recv_buffer->ptr(), m_size, SEED, mem_type());
    }
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type_alloc_before_init, all, "all")


/*
 * Moves a payload out of memory of the tested type, allocated by UCP or by the
 * application, over a rendezvous protocol. The zero-copy mode covers the
 * registration path a network transport uses to reach the memory directly,
 * including registration through a DMA-BUF handle: memory which is registered
 * but not readable by the transport is detected by the payload check rather
 * than silently reported as a successful transfer. The staged mode covers the
 * transfer when the lane cannot register the memory.
 */
class test_ucp_mem_type_rndv_zcopy : public test_ucp_mem_type {
public:
    static const uint64_t SEED = 0x1234567890abcdeflu;

    enum xfer_mode {
        XFER_ZCOPY, /* the transport accesses the source memory directly */
        XFER_STAGED /* the source is not registered on the rendezvous lane */
    };

protected:
    typedef ucs::handle<ucp_mem_h, ucp_context_h> mem_handle_t;

    static void unmap_memh(ucp_mem_h memh, ucp_context_h context)
    {
        ucs_status_t status = ucp_mem_unmap(context, memh);
        if (status != UCS_OK) {
            ucs_warn("failed to unmap memory: %s", ucs_status_string(status));
        }
    }

    void init() override
    {
        test_ucp_mem_type::init();
        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());
    }

    /* Allocate through UCP so that a memory domain performs the allocation */
    void *alloc_mem(entity &e, size_t length, ucs_memory_type_t mem_type,
                    mem_handle_t &memh_handle)
    {
        ucp_mem_map_params_t params;
        ucp_mem_attr_t attr;
        ucp_mem_h memh;
        ucs_status_t status;

        params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                             UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                             UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                             UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        params.address     = NULL;
        params.length      = length;
        params.flags       = UCP_MEM_MAP_ALLOCATE;
        params.memory_type = mem_type;

        status = ucp_mem_map(e.ucph(), &params, &memh);
        if (status != UCS_OK) {
            UCS_TEST_SKIP_R(std::string("cannot allocate ") +
                            ucs_memory_type_names[mem_type] + " memory: " +
                            ucs_status_string(status));
        }

        memh_handle.reset(memh, unmap_memh, e.ucph());

        attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
        status          = ucp_mem_query(memh, &attr);
        if (status != UCS_OK) {
            UCS_TEST_ABORT("ucp_mem_query() failed: " +
                           std::string(ucs_status_string(status)));
        }

        return attr.address;
    }

    /*
     * Memory domains used by the endpoint's rendezvous zero-copy lanes, without
     * the domains used for memory type detection: those register their own
     * memory type regardless of DMA-BUF support.
     *
     * 'remote' returns the destination domains, which is what a GET issued by
     * this endpoint addresses on its peer, so the indices are in the peer's
     * domain space. Otherwise the endpoint's own domains are returned, which is
     * what a PUT issued by this endpoint registers locally.
     */
    ucp_md_map_t rndv_zcopy_md_map(entity &e, bool remote)
    {
        const ucp_ep_config_t *ep_config = ucp_ep_config(e.ep());
        ucp_context_h context            = e.ucph();
        ucp_md_map_t copy_md_map         = 0;
        ucp_md_map_t md_map              = 0;
        ucp_lane_index_t i, lane;
        ucp_rsc_index_t rsc_index;
        ucp_md_index_t md_index;

        for (md_index = 0; md_index < context->num_mem_type_detect_mds;
             ++md_index) {
            copy_md_map |= UCS_BIT(context->mem_type_detect_mds[md_index]);
        }

        for (i = 0; i < UCP_MAX_LANES; ++i) {
            lane = ep_config->key.rma_bw_lanes[i];
            if (lane == UCP_NULL_LANE) {
                break;
            }

            if (remote) {
                md_map |= UCS_BIT(ep_config->key.lanes[lane].dst_md_index);
            } else {
                rsc_index = ep_config->key.lanes[lane].rsc_index;
                md_map   |= UCS_BIT(context->tl_rscs[rsc_index].md_index);
            }
        }

        return md_map & ~copy_md_map;
    }

    std::string md_map_str(entity &e, ucp_md_map_t md_map)
    {
        ucp_context_h context = e.ucph();
        std::string names;
        ucp_md_index_t md_index;

        ucs_for_each_bit(md_index, md_map) {
            if (!names.empty()) {
                names += ",";
            }
            names += (md_index < context->num_mds) ?
                             context->tl_mds[md_index].rsc.md_name :
                             "<unknown>";
        }

        return names.empty() ? "<none>" : names;
    }

    /*
     * Protocol selected for a tagged send of 'length' bytes from 'mem_type'
     * memory, with its description as reported by UCX_PROTO_INFO.
     */
    std::string tag_send_proto(ucs_memory_type_t mem_type, size_t length,
                               std::string &desc)
    {
        ucp_worker_h worker                 = sender().worker();
        ucp_worker_cfg_index_t ep_cfg_index = sender().ep()->cfg_index;
        ucp_memory_info_t mem_info          = {
            .type    = static_cast<uint8_t>(mem_type),
            .sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN,
            .flags   = UCS_MEM_FLAG_REGISTRABLE
        };
        const ucp_proto_threshold_elem_t *thresh;
        ucp_proto_select_param_t select_param;
        ucp_proto_select_elem_t *select_elem;
        ucp_proto_query_attr_t attr;

        ucp_proto_select_param_init(&select_param, UCP_OP_ID_TAG_SEND, 0, 0,
                                    UCP_DATATYPE_CONTIG, &mem_info, 1);
        select_elem = ucp_proto_select_lookup_slow(
                worker,
                &ucs_array_elem(&worker->ep_config, ep_cfg_index).proto_select,
                0, ep_cfg_index, UCP_WORKER_CFG_INDEX_NULL, &select_param);
        if (select_elem == NULL) {
            desc.clear();
            return "";
        }

        thresh = ucp_proto_thresholds_search_slow(select_elem->thresholds,
                                                  length);
        ucp_proto_config_query(worker, &thresh->proto_config, length, &attr);
        desc = attr.desc;
        return thresh->proto_config.proto->name;
    }

    /*
     * Send 'length' bytes of the tested memory type to a host buffer over a
     * rendezvous protocol. 'app_owned' allocates the source without UCP and
     * sends it without a memory handle, like an application would.
     */
    void test_xfer_from_mem_type(size_t length, bool is_get, xfer_mode mode,
                                 bool app_owned);
};

void test_ucp_mem_type_rndv_zcopy::test_xfer_from_mem_type(size_t length,
                                                          bool is_get,
                                                          xfer_mode mode,
                                                          bool app_owned)
{
    const ucs_memory_type_t send_mem_type = mem_type();
    const char *mem_type_name             =
            ucs_memory_type_names[send_mem_type];
    ucp_request_param_t param;
    mem_handle_t send_memh, recv_memh;
    std::unique_ptr<mem_buffer> app_buf;
    std::string proto_name, proto_desc;
    void *send_buf, *recv_buf;
    void *sreq, *rreq;
    bool can_reg;

    if (!mem_buffer::is_mem_type_supported(send_mem_type)) {
        UCS_TEST_SKIP_R(std::string(mem_type_name) +
                        " memory is not supported");
    }

    /*
     * Resolve the lanes and the registration capability before allocating, so
     * that skipping does not leak the memory mappings.
     */
    entity &op_entity        = is_get ? receiver() : sender();
    ucp_md_map_t lane_md_map = rndv_zcopy_md_map(op_entity, is_get);
    if (lane_md_map == 0) {
        UCS_TEST_SKIP_R("no rendezvous zero-copy lane is available");
    }

    can_reg = sender().ucph()->reg_md_map[send_mem_type] & lane_md_map;
    if ((mode == XFER_ZCOPY) && !can_reg) {
        UCS_TEST_SKIP_R(
                std::string("the rendezvous zero-copy lane cannot register ") +
                mem_type_name + " memory");
    } else if ((mode == XFER_STAGED) && can_reg) {
        UCS_TEST_SKIP_R(
                std::string("the rendezvous zero-copy lane can register ") +
                mem_type_name + " memory, nothing to stage");
    }

    /* A forced zero-copy scheme can fall back to eager or to AM rendezvous when
     * it cannot access the source, so check the selected protocol as well as
     * the payload */
    proto_name = tag_send_proto(send_mem_type, length, proto_desc);
    UCS_TEST_MESSAGE << "protocol: " << proto_name << ", " << proto_desc;
    EXPECT_EQ("tag/rndv", proto_name) << "no rendezvous protocol is selected";
    if (mode == XFER_ZCOPY) {
        EXPECT_NE(std::string::npos, proto_desc.find("zero-copy"))
                << "the source memory is not accessed by the transport";
    } else {
        EXPECT_EQ(std::string::npos, proto_desc.find("zero-copy"))
                << "the source memory is accessed by the transport";
        /* CPU-accessible memory is copied by the CPU instead of being staged */
        if (!(UCS_BIT(send_mem_type) & UCS_MEMORY_TYPES_CPU_ACCESSIBLE)) {
            EXPECT_NE(std::string::npos, proto_desc.find("frag host"))
                    << "device memory is not staged through host fragments";
        }
    }

    if (app_owned) {
        app_buf.reset(new mem_buffer(length, send_mem_type));
        send_buf = app_buf->ptr();
    } else {
        send_buf = alloc_mem(sender(), length, send_mem_type, send_memh);
    }
    recv_buf = alloc_mem(receiver(), length, UCS_MEMORY_TYPE_HOST, recv_memh);

    if (!app_owned) {
        UCS_TEST_MESSAGE << (is_get ? "get" : "put") << " lane mds: "
                         << md_map_str(sender(), lane_md_map)
                         << " | source memh mds: "
                         << md_map_str(sender(), send_memh->md_map);

        if (mode == XFER_ZCOPY) {
            EXPECT_NE(0u, send_memh->md_map & lane_md_map)
                    << mem_type_name
                    << " memory is not registered on the memory domain used by "
                       "the rendezvous zero-copy lane, the payload would be "
                       "staged instead of accessed by the transport";
        } else {
            EXPECT_EQ(0u, send_memh->md_map & lane_md_map)
                    << mem_type_name
                    << " memory is registered on the memory domain used by the "
                       "rendezvous zero-copy lane, the payload would be "
                       "accessed by the transport instead of staged";
        }
    }

    mem_buffer::pattern_fill(send_buf, length, SEED, send_mem_type);
    mem_buffer::pattern_fill(recv_buf, length, 0, UCS_MEMORY_TYPE_HOST);

    param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
    param.memh         = recv_memh;
    rreq = ucp_tag_recv_nbx(receiver().worker(), recv_buf, length, 1, 1,
                            &param);

    if (app_owned) {
        param.op_attr_mask = 0;
    } else {
        param.memh = send_memh;
    }
    sreq = ucp_tag_send_nbx(sender().ep(), send_buf, length, 1, &param);

    EXPECT_UCS_OK(request_wait(sreq));
    EXPECT_UCS_OK(request_wait(rreq));

    mem_buffer::pattern_check(recv_buf, length, SEED, UCS_MEMORY_TYPE_HOST);
}

UCS_TEST_P(test_ucp_mem_type_rndv_zcopy, get_zcopy, "RNDV_THRESH=0",
           "RNDV_SCHEME=get_zcopy")
{
    test_xfer_from_mem_type(4 * UCS_MBYTE, true, XFER_ZCOPY, false);
}

UCS_TEST_P(test_ucp_mem_type_rndv_zcopy, put_zcopy, "RNDV_THRESH=0",
           "RNDV_SCHEME=put_zcopy")
{
    test_xfer_from_mem_type(4 * UCS_MBYTE, false, XFER_ZCOPY, false);
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_mem_type_rndv_zcopy, rcx,
                                        "rc_x")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type_rndv_zcopy, rcx_ze,
                              "rc_x,ze_copy")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type_rndv_zcopy, rcv_ze,
                              "rc_v,ze_copy")


/*
 * With DMA-BUF export disabled, the rendezvous zero-copy lane cannot register
 * Level Zero memory and the payload must be copied through host memory: device
 * memory is staged through fragments, CPU-accessible memory is copied in and
 * out. The rendezvous scheme is left to UCP: forcing a zero-copy scheme would
 * exclude these protocols.
 */
class test_ucp_mem_type_rndv_staged : public test_ucp_mem_type_rndv_zcopy {
public:
    test_ucp_mem_type_rndv_staged()
    {
        m_env.push_back(new ucs::scoped_setenv("UCX_ZE_COPY_DMABUF", "n"));
    }
};

UCS_TEST_P(test_ucp_mem_type_rndv_staged, staged, "RNDV_THRESH=0",
           "RNDV_FRAG_MEM_TYPES=host")
{
    test_xfer_from_mem_type(4 * UCS_MBYTE, false, XFER_STAGED, false);
}

UCS_TEST_P(test_ucp_mem_type_rndv_staged, staged_app_mem, "RNDV_THRESH=0",
           "RNDV_FRAG_MEM_TYPES=host")
{
    test_xfer_from_mem_type(4 * UCS_MBYTE, false, XFER_STAGED, true);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type_rndv_staged, rcx_ze,
                              "rc_x,ze_copy")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type_rndv_staged, rcv_ze,
                              "rc_v,ze_copy")

class test_ucp_cuda : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_AM);
    }
};

UCS_TEST_P(test_ucp_cuda, sparse_regions) {
    const ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_CUDA;
    const size_t size = 4096;
    const size_t count = 5;
    ucs_status_t status;
    void *ptr[count];
    ucp_mem_h memh[count];

    if (!mem_buffer::is_mem_type_supported(mem_type)) {
        UCS_TEST_SKIP_R("CUDA is not supported");
    }

    /* create contiguous CUDA registrations list */
    for (int i = 0; i < count; i++) {
        ptr[i] = mem_buffer::allocate(size, mem_type);

        if ((i > 0) && (UCS_PTR_BYTE_OFFSET(ptr[i - 1], size) != ptr[i])) {
            for (int j = 0; j < i; j++) {
                mem_buffer::release(ptr[j], mem_type);
            }
            UCS_TEST_SKIP_R("failed to create contiguous CUDA registrations list");
        }
    }

    /* make CUDA registrations list sparse */
    for (int i = 0; i < count; i++) {
        if ((i & 1) == 0) {
            mem_buffer::release(ptr[i], mem_type);
        }
    }

    for (int i = 0; i < count; i++) {
        if ((i & 1) == 0) {
            continue;
        }

        ucp_mem_map_params_t params;
        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        params.address    = ptr[i];
        params.length     = size;

        status = ucp_mem_map(sender().ucph(), &params, &memh[i]);
        ASSERT_UCS_OK(status);
    }

    for (int i = 0; i < count; i++) {
        if ((i & 1) == 0) {
            continue;
        }

        void* rkey_buffer;
        size_t rkey_buffer_size;
        status = ucp_rkey_pack(sender().ucph(), memh[i], &rkey_buffer,
                               &rkey_buffer_size);
        ASSERT_UCS_OK(status);
        ucp_rkey_buffer_release(rkey_buffer);
    }

    for (int i = 0; i < count; i++) {
        if ((i & 1) == 0) {
            continue;
        }

        status = ucp_mem_unmap(sender().ucph(), memh[i]);
        ASSERT_UCS_OK(status);
    }

    for (int i = 0; i < count; i++) {
        if ((i & 1) == 1) {
            mem_buffer::release(ptr[i], mem_type);
        }
    }
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_cuda, all, "all")

/*
 * Stream-ordered CUDA memory is not GDR capable, so gdr_copy cannot
 * register it. Pack/unpack through the CUDA memory type endpoint has to fall
 * back from the preferred gdr_copy lane to cuda_copy.
 */
class test_ucp_mem_type_non_gdr_capable : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant(variants, UCP_FEATURE_AM);
    }

    virtual void init() override
    {
        /* Type the stream-ordered allocations as plain CUDA memory, so they
         * reach the CUDA memory type endpoint where gdr_copy is selected */
        modify_config("CUDA_COPY_ASYNC_MEM_TYPE", "cuda", SETENV_IF_NOT_EXIST);
        ucp_test::init();
    }

protected:
    static constexpr size_t BUF_SIZE = 65536;
    static constexpr uint64_t SEED   = 0xdeadbeefcafebabeull;

    /* The datatype iterator passes the resolved memory flags, while the
     * protocols which do not carry them pass NULL */
    void test_pack_unpack(int pass_mem_info)
    {
        std::vector<uint8_t> host_buf(BUF_SIZE);
        ucp_memory_info_t mem_info;
        ucp_ep_h mem_type_ep;

        if (!mem_buffer::is_async_supported(UCS_MEMORY_TYPE_CUDA)) {
            UCS_TEST_SKIP_R("CUDA async allocation is not supported");
        }

        mem_type_ep = sender().worker()->mem_type_ep[UCS_MEMORY_TYPE_CUDA];
        if (mem_type_ep == NULL) {
            UCS_TEST_SKIP_R("no CUDA mem type endpoint");
        }

        /* Without a preferred lane which demands memory flags, pack/unpack has
         * nothing to fall back from */
        ucp_lane_index_t lane = ucp_ep_config(mem_type_ep)->key.rma_lanes[0];
        if ((lane == UCP_NULL_LANE) ||
            (ucp_ep_md_attr(mem_type_ep, lane)->required_mem_flags == 0)) {
            UCS_TEST_SKIP_R("first CUDA mem type lane requires no memory flags");
        }

        mem_buffer src(BUF_SIZE, UCS_MEMORY_TYPE_CUDA,
                       mem_buffer::alloc_mode::ASYNC);
        mem_buffer dst(BUF_SIZE, UCS_MEMORY_TYPE_CUDA,
                       mem_buffer::alloc_mode::ASYNC);

        ucp_memory_detect(sender().ucph(), src.ptr(), BUF_SIZE, &mem_info);
        if (mem_info.flags & UCS_MEM_FLAG_GDR_CAPABLE) {
            UCS_TEST_SKIP_R("CUDA async memory is GDR capable");
        }

        ASSERT_EQ(UCS_MEMORY_TYPE_CUDA, mem_info.type);

        const ucp_memory_info_t *mem_info_arg = pass_mem_info ? &mem_info :
                                                                NULL;

        mem_buffer::pattern_fill(src.ptr(), BUF_SIZE, SEED,
                                 UCS_MEMORY_TYPE_CUDA);
        mem_buffer::pattern_fill(dst.ptr(), BUF_SIZE, 0, UCS_MEMORY_TYPE_CUDA);

        ucp_mem_type_pack(sender().worker(), host_buf.data(), src.ptr(),
                          BUF_SIZE, UCS_MEMORY_TYPE_CUDA, mem_info_arg);
        mem_buffer::pattern_check(host_buf.data(), BUF_SIZE, SEED);

        ucp_mem_type_unpack(sender().worker(), dst.ptr(), host_buf.data(),
                            BUF_SIZE, UCS_MEMORY_TYPE_CUDA, mem_info_arg);
        mem_buffer::pattern_check(dst.ptr(), BUF_SIZE, SEED,
                                  UCS_MEMORY_TYPE_CUDA);
    }
};

UCS_TEST_P(test_ucp_mem_type_non_gdr_capable, pack_unpack_resolved_flags)
{
    test_pack_unpack(1);
}

/* Without the flags from the caller, they are resolved from the buffer address
 * on the first lane whose memory domain requires them */
UCS_TEST_P(test_ucp_mem_type_non_gdr_capable, pack_unpack_unknown_flags)
{
    test_pack_unpack(0);
}

/* Resolving the flags hits the memtype cache first, make sure the slow path
 * also reports them when the cache is disabled */
UCS_TEST_P(test_ucp_mem_type_non_gdr_capable, pack_unpack_no_memtype_cache,
           "MEMTYPE_CACHE=n")
{
    test_pack_unpack(0);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type_non_gdr_capable, all, "all")
