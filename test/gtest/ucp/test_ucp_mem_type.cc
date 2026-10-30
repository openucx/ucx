/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
* Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"
#include <common/mem_buffer.h>

#include <algorithm>
#include <map>
#include <memory>

extern "C" {
#include <uct/api/uct.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/proto/proto_select.h>
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
    static constexpr uint64_t SEED = 0x1234567890abcdeflu;

    enum xfer_mode {
        XFER_ZCOPY, /* the transport accesses the source memory directly */
        XFER_STAGED /* the source is not registered on the rendezvous lane */
    };

protected:
    using mem_handle_t = ucs::handle<ucp_mem_h, ucp_context_h>;

    static void unmap_memh(ucp_mem_h memh, ucp_context_h context)
    {
        ucs_status_t status = ucp_mem_unmap(context, memh);
        if (status != UCS_OK) {
            ucs_warn("failed to unmap memory: %s", ucs_status_string(status));
        }
    }

    void init() override
    {
        /* Count protocol selections, which used_protos() reports */
        modify_config("PROTO_INFO", "used");
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

    struct proto_selections_t {
        ucp_worker_h      worker;
        ucs_memory_type_t remote_mem_type; /* UNKNOWN without a remote key */
        unsigned          selections;
    };

    using proto_counts_t = std::map<const ucp_proto_config_t*,
                                    proto_selections_t>;

    struct used_proto_t {
        std::string       name;
        std::string       desc;
        bool              is_estimation;   /* desc predicts the peer's part */
        ucs_memory_type_t mem_type;        /* Local buffer */
        ucs_memory_type_t remote_mem_type; /* Remote key, or UNKNOWN */
    };

    static void add_selections(ucp_worker_h worker,
                               const ucp_proto_select_t *proto_select,
                               ucs_memory_type_t remote_mem_type,
                               proto_counts_t &counts)
    {
        const ucp_proto_threshold_elem_t *thresh;
        khiter_t khiter;

        for (khiter = kh_begin(proto_select->hash);
             khiter != kh_end(proto_select->hash); ++khiter) {
            if (!kh_exist(proto_select->hash, khiter)) {
                continue;
            }

            thresh = kh_val(proto_select->hash, khiter).thresholds;
            do {
                counts[&thresh->proto_config] = {
                    worker, remote_mem_type, thresh->proto_config.selections
                };
            } while ((thresh++)->max_msg_length < SIZE_MAX);
        }
    }

    /*
     * Selection counters of every protocol configured on the sender and the
     * receiver, which count requests only with PROTO_INFO=used.
     */
    proto_counts_t proto_selections()
    {
        proto_counts_t counts;
        ucp_rkey_config_t **rkey_config_p;
        ucp_ep_config_t *ep_config;

        for (entity *e : {&sender(), &receiver()}) {
            ucp_worker_h worker = e->worker();

            ucs_array_for_each(ep_config, &worker->ep_config) {
                add_selections(worker, &ep_config->proto_select,
                               UCS_MEMORY_TYPE_UNKNOWN, counts);
            }

            ucs_array_for_each(rkey_config_p, &worker->rkey_config) {
                add_selections(worker, &(*rkey_config_p)->proto_select,
                               (*rkey_config_p)->key.mem_type, counts);
            }
        }

        return counts;
    }

    /*
     * Protocols which started a request since 'before' was taken, including
     * the rendezvous protocols which move the data after the tag/rndv request
     * is matched.
     */
    std::vector<used_proto_t>
    used_protos(const proto_counts_t &before, size_t length)
    {
        std::vector<used_proto_t> used;
        ucp_proto_query_attr_t attr;

        for (const auto &count : proto_selections()) {
            auto prev          = before.find(count.first);
            unsigned prev_sels = (prev == before.end()) ?
                                         0 : prev->second.selections;
            if (count.second.selections <= prev_sels) {
                continue;
            }

            ucp_proto_config_query(count.second.worker, count.first, length,
                                   &attr);
            used.push_back({count.first->proto->name, attr.desc,
                            static_cast<bool>(attr.is_estimation),
                            static_cast<ucs_memory_type_t>(
                                    count.first->select_param.mem_type),
                            count.second.remote_mem_type});
        }

        return used;
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
    mem_handle_t pad_memh, send_memh, recv_memh;
    std::unique_ptr<mem_buffer> app_buf;
    proto_counts_t sels_before;
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

    if (app_owned) {
        app_buf.reset(new mem_buffer(length, send_mem_type));
        send_buf = app_buf->ptr();
    } else {
        /* Keep another buffer of the same size allocated, so that a driver
         * which sub-allocates both from one memory pool places the source at a
         * nonzero offset in the pool, where a DMA-BUF exported for an
         * allocation not created for export does not describe the source */
        alloc_mem(sender(), length, send_mem_type, pad_memh);
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

    sels_before = proto_selections();

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

    /* A forced zero-copy scheme can fall back to eager or to AM rendezvous when
     * it cannot access the source, so check the protocols which moved the data
     * as well as the payload */
    std::vector<used_proto_t> used = used_protos(sels_before, length);
    std::string used_str;
    for (const used_proto_t &proto : used) {
        used_str += std::string(" [") + proto.name + ": " +
                    (proto.is_estimation ? "(?) " : "") + proto.desc +
                    ", local " + ucs_memory_type_names[proto.mem_type] +
                    ", remote " + ucs_memory_type_names[proto.remote_mem_type] +
                    "]";
    }
    UCS_TEST_MESSAGE << "used protocols:" << used_str;

    /* A zero-copy protocol on the source memory: a PUT or an AM send reads
     * its local buffer, a GET issued by the receiver reads the remote one */
    auto is_used_on_source = [&](const std::string &name) {
        return std::any_of(used.begin(), used.end(),
                           [&](const used_proto_t &proto) {
                               return (proto.name == name) &&
                                      ((proto.mem_type == send_mem_type) ||
                                       (proto.remote_mem_type ==
                                        send_mem_type));
                           });
    };

    if (mode == XFER_ZCOPY) {
        EXPECT_TRUE(is_used_on_source(is_get ? "rndv/get/zcopy" :
                                               "rndv/put/zcopy"))
                << "the source memory is not accessed by the transport";
    } else {
        EXPECT_TRUE(is_used_on_source("tag/rndv"))
                << "the source memory is not sent by a rendezvous protocol";
        EXPECT_FALSE(is_used_on_source("rndv/get/zcopy") ||
                     is_used_on_source("rndv/put/zcopy") ||
                     is_used_on_source("rndv/am/zcopy"))
                << "the source memory is accessed by the transport";
        /* CPU-accessible memory may be copied by the CPU instead of being
         * staged. The rendezvous request and reply only estimate the peer's
         * protocol, so skip estimated descriptions. */
        if (!(UCS_BIT(send_mem_type) & UCS_MEMORY_TYPES_CPU_ACCESSIBLE)) {
            EXPECT_TRUE(std::any_of(used.begin(), used.end(),
                                    [](const used_proto_t &proto) {
                                        return !proto.is_estimation &&
                                               (proto.desc.find("frag host") !=
                                                std::string::npos);
                                    }))
                    << "device memory is not staged through host fragments";
        }
    }
}

/* Small enough for a driver to sub-allocate from a memory pool */
static constexpr size_t RNDV_ZCOPY_LENGTH = 64 * UCS_KBYTE;

UCS_TEST_P(test_ucp_mem_type_rndv_zcopy, get_zcopy, "RNDV_THRESH=0",
           "RNDV_SCHEME=get_zcopy")
{
    test_xfer_from_mem_type(RNDV_ZCOPY_LENGTH, true, XFER_ZCOPY, false);
}

UCS_TEST_P(test_ucp_mem_type_rndv_zcopy, put_zcopy, "RNDV_THRESH=0",
           "RNDV_SCHEME=put_zcopy")
{
    test_xfer_from_mem_type(RNDV_ZCOPY_LENGTH, false, XFER_ZCOPY, false);
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
 * memory is staged through fragments, CPU-accessible memory is staged or copied
 * in and out. The rendezvous scheme is left to UCP: forcing a zero-copy scheme
 * would exclude these protocols.
 */
class test_ucp_mem_type_rndv_staged : public test_ucp_mem_type_rndv_zcopy {
public:
    test_ucp_mem_type_rndv_staged()
    {
        m_env.push_back(new ucs::scoped_setenv("UCX_ZE_COPY_DMABUF", "n"));
    }

    /* Disabling DMA-BUF export changes how only Level Zero memory is reached */
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        static constexpr uint64_t ze_mem_types =
                UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE) |
                UCS_BIT(UCS_MEMORY_TYPE_ZE_MANAGED);

        add_variant_memtypes(variants, get_test_variants_base, ze_mem_types);
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
