/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"

#include <common/test.h>
#include <common/mem_buffer.h>
#include <unordered_map>
#include <memory>

extern "C" {
#include <ucp/core/ucp_rkey.h>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_perf.h>
#include <ucp/proto/proto_init.h>
#include <ucs/datastruct/linear_func.h>
#include <ucp/proto/proto_select.inl>
#include <ucp/core/ucp_worker.inl>
}

class test_ucp_proto : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_TAG | UCP_FEATURE_RMA);
    }

protected:
    void do_mem_reg(ucp_datatype_iter_t *dt_iter, ucp_md_map_t md_map);

    ucp_md_map_t get_md_map(ucs_memory_type_t mem_type);

    void test_dt_iter_mem_reg(ucs_memory_type_t mem_type, size_t size,
                              ucp_md_map_t md_map);

    virtual void init() {
        modify_config("PROTO_ENABLE", "y");
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
    }

    ucp_context_h context() {
        return sender().ucph();
    }

    ucp_worker_h worker() {
        return sender().worker();
    }

    static ucp_rkey_config_key_t create_rkey_config_key(ucp_md_map_t md_map);
};

ucp_md_map_t test_ucp_proto::get_md_map(ucs_memory_type_t mem_type)
{
    return context()->reg_md_map[mem_type] &
    /* ucp_datatype_iter_mem_reg() always goes directly to registration cache */
           context()->cache_md_map[mem_type];
}

void test_ucp_proto::do_mem_reg(ucp_datatype_iter_t *dt_iter,
                                ucp_md_map_t md_map)
{
    ucp_datatype_iter_mem_reg(context(), dt_iter, md_map, UCT_MD_MEM_ACCESS_ALL,
                              UCP_DT_MASK_ALL);
    ucp_datatype_iter_mem_dereg(dt_iter, UCP_DT_MASK_ALL);
}

void test_ucp_proto::test_dt_iter_mem_reg(ucs_memory_type_t mem_type,
                                          size_t size, ucp_md_map_t md_map)
{
    const double test_time_sec = 1.0;
    mem_buffer buffer(size, mem_type);

    ucp_datatype_iter_t dt_iter;
    uint8_t sg_count;
    /* Pass empty param argument to disable memh initialization */
    ucp_request_param_t param;
    param.op_attr_mask = 0;

    ucp_datatype_iter_init(context(), buffer.ptr(), size, UCP_DATATYPE_CONTIG,
                           size, 1, &dt_iter, &sg_count, &param);

    ucs_time_t start_time = ucs_get_time();
    ucs_time_t deadline   = start_time + ucs_time_from_sec(test_time_sec);
    ucs_time_t end_time   = start_time;
    unsigned count        = 0;
    do {
        do_mem_reg(&dt_iter, md_map);
        ++count;
        if ((count % 8) == 0) {
            end_time = ucs_get_time();
        }
    } while (end_time < deadline);

    char memunits_str[32];
    UCS_TEST_MESSAGE << ucs_memory_type_names[mem_type] << " "
                     << ucs_memunits_to_str(size, memunits_str,
                                            sizeof(memunits_str))
                     << " md_map 0x" << std::hex << md_map << std::dec
                     << " registration time: "
                     << (ucs_time_to_nsec(end_time - start_time) / count)
                     << " nsec";
}

ucp_rkey_config_key_t
test_ucp_proto::create_rkey_config_key(ucp_md_map_t md_map)
{
    ucp_rkey_config_key_t rkey_config_key;

    rkey_config_key.ep_cfg_index       = 0;
    rkey_config_key.md_map             = md_map;
    rkey_config_key.mem_type           = UCS_MEMORY_TYPE_HOST;
    rkey_config_key.sys_dev            = UCS_SYS_DEVICE_ID_UNKNOWN;
    rkey_config_key.unreachable_md_map = 0;

    return rkey_config_key;
}

UCS_TEST_P(test_ucp_proto, dump_protocols) {
    ucp_proto_select_param_t select_param;
    ucs_string_buffer_t strb;

    select_param.op_id_flags   = UCP_OP_ID_TAG_SEND;
    select_param.op_attr       = 0;
    select_param.dt_class      = UCP_DATATYPE_CONTIG;
    select_param.mem_type      = UCS_MEMORY_TYPE_HOST;
    select_param.sys_dev       = UCS_SYS_DEVICE_ID_UNKNOWN;
    select_param.sg_count      = 1;
    select_param.op.padding[0] = 0;
    select_param.op.padding[1] = 0;

    ucs_string_buffer_init(&strb);
    ucp_proto_select_param_str(&select_param, ucp_operation_names, &strb);
    UCS_TEST_MESSAGE << ucs_string_buffer_cstr(&strb);
    ucs_string_buffer_cleanup(&strb);

    ucp_worker_h worker                   = sender().worker();
    ucp_worker_cfg_index_t ep_cfg_index   = sender().ep()->cfg_index;
    ucp_worker_cfg_index_t rkey_cfg_index = UCP_WORKER_CFG_INDEX_NULL;

    auto proto_select = &ucs_array_elem(&worker->ep_config,
                                        ep_cfg_index).proto_select;
    auto select_elem  = ucp_proto_select_lookup(worker, proto_select,
                                                ep_cfg_index, rkey_cfg_index,
                                                &select_param, 0);
    EXPECT_NE(nullptr, select_elem);

    ucp_ep_print_info(sender().ep(), stdout);
}

UCS_TEST_P(test_ucp_proto, rkey_config) {
    ucp_rkey_config_key_t rkey_config_key = create_rkey_config_key(0);
    ucs_status_t status;

    /* similar configurations should return same index */
    ucp_worker_cfg_index_t cfg_index1;
    status = ucp_worker_rkey_config_get(worker(), &rkey_config_key, NULL,
                                        &cfg_index1);
    ASSERT_UCS_OK(status);

    ucp_worker_cfg_index_t cfg_index2;
    status = ucp_worker_rkey_config_get(worker(), &rkey_config_key, NULL,
                                        &cfg_index2);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(static_cast<int>(cfg_index1), static_cast<int>(cfg_index2));

    rkey_config_key = create_rkey_config_key(1);

    /* different configuration should return different index */
    ucp_worker_cfg_index_t cfg_index3;
    status = ucp_worker_rkey_config_get(worker(), &rkey_config_key, NULL,
                                        &cfg_index3);
    ASSERT_UCS_OK(status);

    EXPECT_NE(static_cast<int>(cfg_index1), static_cast<int>(cfg_index3));
}

UCS_TEST_P(test_ucp_proto, worker_print_info_rkey)
{
    ucp_rkey_config_key_t rkey_config_key = create_rkey_config_key(0);

    ucp_worker_cfg_index_t cfg_index;
    ucs_status_t status = ucp_worker_rkey_config_get(worker(), &rkey_config_key,
                                                     NULL, &cfg_index);
    ASSERT_UCS_OK(status);

    ucp_worker_print_info(worker(), stdout);
}

UCS_TEST_P(test_ucp_proto, dt_iter_mem_reg)
{
    static const size_t buffer_size = 8192;

    for (auto mem_type : mem_buffer::supported_mem_types()) {
        ucp_md_map_t md_map = get_md_map(mem_type);
        if (md_map == 0) {
            UCS_TEST_MESSAGE << "No memory domains can register "
                             << ucs_memory_type_names[mem_type] << " memory";
            continue;
        }

        test_dt_iter_mem_reg(mem_type, buffer_size, md_map);
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_proto)
UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_proto, shm_ipc,
                                        "shm,cuda_ipc,rocm_ipc")

class test_perf_node : public test_ucp_proto {
};

UCS_TEST_P(test_perf_node, basic)
{
    static const std::string nullstr = "(null)";

    EXPECT_EQ(nullstr, ucp_proto_perf_node_name(NULL));
    EXPECT_EQ(nullstr, ucp_proto_perf_node_desc(NULL));

    ucp_proto_perf_node_t *n1 = ucp_proto_perf_node_new_data("n1", "node%d", 1);
    ASSERT_NE(nullptr, n1);
    EXPECT_EQ(std::string("n1"), ucp_proto_perf_node_name(n1));
    EXPECT_EQ(std::string("node1"), ucp_proto_perf_node_desc(n1));

    ucp_proto_perf_node_t *n2 = ucp_proto_perf_node_new_select("n2", 0,
                                                               "node%d", 2);
    ASSERT_NE(nullptr, n2);
    EXPECT_EQ(std::string("n2"), ucp_proto_perf_node_name(n2));
    EXPECT_EQ(std::string("node2"), ucp_proto_perf_node_desc(n2));

    ucp_proto_perf_node_t *n3 = ucp_proto_perf_node_new_data("n3", "node%d", 3);
    ASSERT_NE(nullptr, n3);
    EXPECT_EQ(std::string("n3"), ucp_proto_perf_node_name(n3));
    EXPECT_EQ(std::string("node3"), ucp_proto_perf_node_desc(n3));

    ucp_proto_perf_node_add_data(n3, "zero", UCS_LINEAR_FUNC_ZERO);
    ucp_proto_perf_node_add_data(n3, "one", ucs_linear_func_make(0, 1));
    ucp_proto_perf_node_add_scalar(n3, "lat", 1e-6);
    ucp_proto_perf_node_add_bandwidth(n3, "bw", UCS_MBYTE);

    /* NULL child is ignored */
    ucp_proto_perf_node_add_child(n1, NULL);
    ucp_proto_perf_node_add_child(n2, NULL);

    ucp_proto_perf_node_t *tmp = NULL;
    ucp_proto_perf_node_own_child(n1, &tmp);

    /* NULL parent is ignored */
    ucp_proto_perf_node_add_child(NULL, n1);
    ucp_proto_perf_node_add_child(NULL, n2);
    ucp_proto_perf_node_add_child(NULL, NULL);

    /* NULL owner should remove extra ref */
    ucp_proto_perf_node_t *n2_ref = n2;
    ucp_proto_perf_node_ref(n2_ref);
    ucp_proto_perf_node_own_child(NULL, &n2_ref);
    EXPECT_EQ(nullptr, n2_ref);

    /* NULL node is ignored */
    ucp_proto_perf_node_add_data(NULL, "ignored", UCS_LINEAR_FUNC_ZERO);
    ucp_proto_perf_node_add_scalar(NULL, "ignored", 1.0);

    /* n1 -> n2 -> n3 */
    ucp_proto_perf_node_own_child(n2, &n3); /* Dropped extra ref to n3 */
    ucp_proto_perf_node_add_child(n1, n2);  /* Have 2 references to n2 */

    ucp_proto_perf_node_deref(&n2); /* n3 should still be alive */
    EXPECT_EQ(nullptr, n2);
    EXPECT_EQ(std::string("node3"), ucp_proto_perf_node_desc(n3));

    ucp_proto_perf_node_deref(&n1); /* Release n1,n2,n3 */
    EXPECT_EQ(nullptr, n1);
}

UCS_TEST_P(test_perf_node, replace_node)
{
    ucp_proto_perf_node_t *parent1 = ucp_proto_perf_node_new_data("parent1", "");
    ucp_proto_perf_node_t *child1 = ucp_proto_perf_node_new_data("child1", "");
    ucp_proto_perf_node_t *child2 = ucp_proto_perf_node_new_data("child2", "");

    ucp_proto_perf_node_add_child(parent1, child1);
    ucp_proto_perf_node_add_child(parent1, child2);
    EXPECT_EQ(child1, ucp_proto_perf_node_get_child(parent1, 0));
    EXPECT_EQ(child2, ucp_proto_perf_node_get_child(parent1, 1));

    ucp_proto_perf_node_t *parent2 = ucp_proto_perf_node_new_data("parent2", "");
    ucp_proto_perf_node_t *parent2_ptr = parent2;
    ASSERT_NE(parent2_ptr, parent1);

    ucp_proto_perf_node_replace(&parent1, &parent2);

    /* parent2 variable should be set to NULL */
    EXPECT_EQ(nullptr, parent2);

    /* parent1 variable should be set to parent2 */
    EXPECT_EQ(parent2_ptr, parent1);

    /* Children should be reassigned */
    EXPECT_EQ(child1, ucp_proto_perf_node_get_child(parent2_ptr, 0));
    EXPECT_EQ(child2, ucp_proto_perf_node_get_child(parent2_ptr, 1));

    ucp_proto_perf_node_deref(&parent1);
    ucp_proto_perf_node_deref(&child1);
    ucp_proto_perf_node_deref(&child2);
}


UCS_TEST_P(test_perf_node, replace_null)
{
    /* Replacing NULL with NULL is a no-op */
    {
        ucp_proto_perf_node_t *n1 = nullptr;
        ucp_proto_perf_node_t *n2 = nullptr;
        ucp_proto_perf_node_replace(&n1, &n2);
    }

    /* Replacing a node by NULL should release the node (without leaks) */
    {
        ucp_proto_perf_node_t *parent1 = ucp_proto_perf_node_new_data("parent1", "");
        ucp_proto_perf_node_t *child1 = ucp_proto_perf_node_new_data("child1", "");
        ucp_proto_perf_node_t *child2 = ucp_proto_perf_node_new_data("child2", "");
        ucp_proto_perf_node_own_child(parent1, &child1);
        ucp_proto_perf_node_own_child(parent1, &child2);

        ucp_proto_perf_node_t *null_node = nullptr;
        ucp_proto_perf_node_replace(&parent1, &null_node);
        EXPECT_EQ(nullptr, parent1);
    }

    /* Replacing a NULL should swap variables */
    {
        ucp_proto_perf_node_t *node = ucp_proto_perf_node_new_data("node", "");
        ucp_proto_perf_node_t *null_node = nullptr;
        ucp_proto_perf_node_replace(&null_node, &node);
        EXPECT_EQ(nullptr, node);
        EXPECT_NE(nullptr, null_node);
        ucp_proto_perf_node_deref(&null_node);
    }
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_perf_node, all, "all")

static std::ostream &operator<<(std::ostream &os, const ucp_proto_perf_t *perf)
{
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    ucp_proto_perf_str(perf, &strb);
    auto &ret = os << ucs_string_buffer_cstr(&strb);
    ucs_string_buffer_cleanup(&strb);
    return ret;
}

static std::ostream &
operator<<(std::ostream &os, const ucp_proto_flat_perf_t *flat_perf)
{
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    ucp_proto_flat_perf_str(flat_perf, &strb);
    auto &ret = os << ucs_string_buffer_cstr(&strb);
    ucs_string_buffer_cleanup(&strb);
    return ret;
}

static std::ostream &operator<<(std::ostream &os, const ucs_linear_func_t &func)
{
    char buffer[128];

    ucs_snprintf_safe(buffer, sizeof(buffer), UCP_PROTO_PERF_FUNC_FMT,
                      UCP_PROTO_PERF_FUNC_ARG(&func));
    return os << buffer;
}

class test_proto_perf : public ucs::test {
public:
    using perf_ptr_t         = std::shared_ptr<ucp_proto_perf_t>;
    using flat_perf_ptr_t    = std::shared_ptr<ucp_proto_flat_perf_t>;
    using perf_factors_map_t = std::unordered_map<int, ucs_linear_func_t>;
protected:
    virtual void cleanup() override
    {
        m_envelope_flat_perf.reset();
        m_sum_flat_perf.reset();
        m_perf.reset();
    }

    static perf_ptr_t create()
    {
        ucp_proto_perf_t *perf;
        ASSERT_UCS_OK(ucp_proto_perf_create("test", &perf));
        return make_perf_ptr(perf);
    }

    perf_ptr_t aggregate(const std::vector<perf_ptr_t> &perfs)
    {
        std::vector<ucp_proto_perf_t*> perfs_ptr_vec;
        std::transform(perfs.begin(), perfs.end(),
                       std::back_inserter(perfs_ptr_vec),
                       [](const perf_ptr_t &p) { return p.get(); });

        ucp_proto_perf_t *perf;
        ASSERT_UCS_OK(ucp_proto_perf_aggregate("aggregate",
                                               perfs_ptr_vec.data(),
                                               perfs_ptr_vec.size(), &perf));
        return make_perf_ptr(perf);
    }

    static void add_funcs(perf_ptr_t perf, size_t start, size_t end,
                          const perf_factors_map_t &factors)
    {
        ucp_proto_perf_factors_t perf_factors =
                UCP_PROTO_PERF_FACTORS_INITIALIZER;

        for (auto &f : factors) {
            perf_factors[f.first] = f.second;
        }
        ASSERT_UCS_OK(ucp_proto_perf_add_funcs(perf.get(), start, end,
                                               perf_factors, NULL, "test", ""));
    }

    static void add_func(perf_ptr_t perf, size_t start, size_t end,
                         ucp_proto_perf_factor_id_t factor_id,
                         ucs_linear_func_t func)
    {
        add_funcs(perf, start, end, {{factor_id, func}});
    }

    void add_func(size_t start, size_t end,
                  ucp_proto_perf_factor_id_t factor_id, ucs_linear_func_t func)
    {
        add_func(m_perf, start, end, factor_id, func);
    }

    static ucp_proto_perf_segment_t *find_lb(perf_ptr_t perf, size_t start)
    {
        return ucp_proto_perf_find_segment_lb(perf.get(), start);
    }

    static const ucp_proto_flat_perf_range_t *
    find_lb(flat_perf_ptr_t flat_perf, size_t start)
    {
        return ucp_proto_flat_perf_find_lb(flat_perf.get(), start);
    }

    void expect_empty_range(size_t start, size_t end) const
    {
        expect_perf_empty_range(m_perf, start, end);
        expect_perf_empty_range(m_sum_flat_perf, start, end);
        expect_perf_empty_range(m_envelope_flat_perf, start, end);
    }

    void expect_perf(size_t start, size_t end,
                     const perf_factors_map_t &factors) const
    {
        ASSERT_FALSE(ucp_proto_perf_is_empty(m_perf.get()));

        const ucp_proto_perf_segment_t *seg = find_lb(m_perf, start);
        ASSERT_NE(nullptr, seg) << "start=" << start;

        EXPECT_GE(start, ucp_proto_perf_segment_start(seg));
        EXPECT_LE(end, ucp_proto_perf_segment_end(seg));

        int factor_id = UCP_PROTO_PERF_FACTOR_LOCAL_CPU;
        for (; factor_id < UCP_PROTO_PERF_FACTOR_LAST; ++factor_id) {
            ucs_linear_func_t expected_func = UCS_LINEAR_FUNC_ZERO;
            auto entry                      = factors.find(factor_id);
            if (entry != std::end(factors)) {
                expected_func = entry->second;
            }

            auto segment_func = ucp_proto_perf_segment_func(
                    seg, (ucp_proto_perf_factor_id_t)factor_id);
            EXPECT_TRUE(
                    ucs_linear_func_is_equal(expected_func, segment_func, 1e-9))
                    << "start=" << start << " end=" << end
                    << " factor_id=" << factor_id
                    << " expected_func=" << expected_func
                    << " segment_func=" << segment_func;
        }

        expect_flat_perf(start, end, factors);
    }

    void make_flat_perf()
    {
        ucp_proto_flat_perf_t *envelope_perf, *sum_perf;

        ASSERT_NE(m_perf.get(), nullptr) << "Perf should be initialized";
        ASSERT_UCS_OK(ucp_proto_perf_envelope(m_perf.get(), 0, &envelope_perf));
        ASSERT_UCS_OK(ucp_proto_perf_sum(m_perf.get(), &sum_perf));

        m_envelope_flat_perf = make_flat_perf_ptr(envelope_perf);
        m_sum_flat_perf      = make_flat_perf_ptr(sum_perf);
    }

    static ucs_linear_func_t perf_func(double overhead_nsec, double bw_mbs)
    {
        return {overhead_nsec * 1e-9, 1.0 / (bw_mbs * UCS_MBYTE)};
    }

    void print_perf() const
    {
        UCS_TEST_MESSAGE << "perf:     " << m_perf.get();
        UCS_TEST_MESSAGE << "envelope: " << m_envelope_flat_perf.get();
        UCS_TEST_MESSAGE << "sum:      " << m_sum_flat_perf.get();
    }

    static perf_ptr_t make_perf_ptr(ucp_proto_perf_t *perf)
    {
        return {perf, ucp_proto_perf_destroy};
    }

private:
    static size_t get_start(ucp_proto_perf_segment_t *seg)
    {
        return ucp_proto_perf_segment_start(seg);
    }

    static size_t get_start(const ucp_proto_flat_perf_range_t *range)
    {
        return range->start;
    }

    template<typename PerfPtrType>
    void
    expect_perf_empty_range(PerfPtrType perf, size_t start, size_t end) const
    {
        ASSERT_NE(perf.get(), nullptr);
        auto *seg = find_lb(perf, start);

        EXPECT_TRUE((seg == nullptr) || (end < get_start(seg)))
                << "perf=" << perf.get() << " start=" << start
                << " end=" << end;
    }

    void expect_flat_range(flat_perf_ptr_t flat_perf, size_t start, size_t end,
                           ucs_linear_func_t exp_func) const
    {
        auto *range     = find_lb(flat_perf, start);
        size_t midpoint = start + (end - start) / 2;

        std::stringstream info;
        info << "expected=[" << start << ", " << end << ", " << exp_func
             << "] actual=[" << range->start << ", " << range->end << ", "
             << range->value << "]";

        EXPECT_NE(range, nullptr) << info.str();
        EXPECT_GE(start, range->start) << info.str();
        EXPECT_LE(end, range->end) << info.str();

        // Cannot compare functions directly since on big sizes (e.g. SIZE_MAX)
        // small latency difference can be overhelmed by floating point math
        // innacurracy after addition with huge BW*SIZE_MAX.
        for (size_t point : {start, midpoint, end}) {
            double expected_result = ucs_linear_func_apply(exp_func, point);
            double range_result = ucs_linear_func_apply(range->value, point);
            EXPECT_NEAR(expected_result, range_result, 10e-9) << info.str();
        }
    }

    void expect_envelope_flat_perf(size_t start, size_t end,
                                   const perf_factors_map_t &factors) const
    {
        ucp_proto_perf_envelope_t envelope = UCS_ARRAY_DYNAMIC_INITIALIZER;
        ucp_proto_perf_envelope_elem_t *envelope_elem;
        std::vector<ucs_linear_func_t> factors_vec;

        for (const auto &factor : factors) {
            if (factor.first == UCP_PROTO_PERF_FACTOR_LATENCY) {
                // Latency shouldn't be considered during envelope building
                factors_vec.emplace_back(UCS_LINEAR_FUNC_ZERO);
            } else {
                factors_vec.emplace_back(factor.second);
            }
        }
        ASSERT_FALSE(factors_vec.empty());

        ASSERT_UCS_OK(ucp_proto_perf_envelope_make(&factors_vec.front(),
                                                   factors_vec.size(), start,
                                                   end, 0, &envelope));

        ucs_array_for_each(envelope_elem, &envelope) {
            ucs_assert(envelope_elem != NULL); /* For coverity */
            expect_flat_range(m_envelope_flat_perf, start,
                              envelope_elem->max_length,
                              factors_vec[envelope_elem->index]);
            start = envelope_elem->max_length + 1;
        }

        ucs_array_cleanup_dynamic(&envelope);
    }

    void expect_sum_flat_perf(size_t start, size_t end,
                              const perf_factors_map_t &factors) const
    {
        ucs_linear_func_t expected_func = UCS_LINEAR_FUNC_ZERO;
        for (const auto &factor : factors) {
            ucs_linear_func_add_inplace(&expected_func, factor.second);
        }

        expect_flat_range(m_sum_flat_perf, start, end, expected_func);
    }

    void expect_flat_perf(size_t start, size_t end,
                          const perf_factors_map_t &factors) const
    {
        // Reduce testing range to `max_envelope_check_size` since precise
        // testing of big sizes is impossible due to floating point loss
        if (start > max_envelope_check_size) {
            return;
        }
        end = std::min(end, max_envelope_check_size);

        expect_envelope_flat_perf(start, end, factors);
        expect_sum_flat_perf(start, end, factors);
    }

    static flat_perf_ptr_t make_flat_perf_ptr(ucp_proto_flat_perf_t *flat_perf)
    {
        return {flat_perf, ucp_proto_flat_perf_destroy};
    }

protected:
    static const size_t            max_envelope_check_size;
    static const ucs_linear_func_t local_tl_func;
    static const ucs_linear_func_t remote_tl_func;
    static const ucs_linear_func_t local_cpu_func;
    flat_perf_ptr_t                m_envelope_flat_perf;
    flat_perf_ptr_t                m_sum_flat_perf;
    perf_ptr_t                     m_perf;
};

const ucs_linear_func_t test_proto_perf::local_tl_func  = perf_func(10, 1000);
const ucs_linear_func_t test_proto_perf::remote_tl_func = perf_func(20, 2000);
const ucs_linear_func_t test_proto_perf::local_cpu_func = perf_func(30, 3000);
const size_t test_proto_perf::max_envelope_check_size   = UCS_GBYTE;

UCS_TEST_F(test_proto_perf, empty)
{
    m_perf = create();

    make_flat_perf();
    print_perf();

    expect_empty_range(0, SIZE_MAX);

    ASSERT_TRUE(ucp_proto_perf_is_empty(m_perf.get()));
}

UCS_TEST_F(test_proto_perf, single_func)
{
    m_perf = create();
    add_func(1000, 1999, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);

    make_flat_perf();
    print_perf();

    expect_empty_range(0, 999);
    expect_perf(1000, 1999, {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_empty_range(2000, SIZE_MAX);
}

UCS_TEST_F(test_proto_perf, to_inf) {
    /*
     *  0    172                   SIZE_MAX
     *  |     |                       |
     *  |     +------ local_tl -------+
     *  |                             |
     *  |                             |
     *  +----------- local_cpu -------+
     */
    m_perf = create();
    add_func(172, SIZE_MAX, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);
    add_func(0, SIZE_MAX, UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func);

    make_flat_perf();
    print_perf();

    expect_perf(0, 171, {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
    expect_perf(172, SIZE_MAX,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func},
                 {UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
}

UCS_TEST_F(test_proto_perf, intersect_first)
{
    /*
     * 500   1000          1999    3000          3999  4999
     *  |     |              |      |              |    |
     *  +-----+-- local_tl --+      +- remote_tl --+    |
     *        |                                         |
     *        |                                         |
     *        +------ local_cpu ------------------------+
     */
    m_perf = create();
    add_func(500, 1999, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);
    add_func(3000, 3999, UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func);
    add_func(1000, 4999, UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func);

    make_flat_perf();
    print_perf();

    expect_empty_range(0, 499);
    expect_perf(500, 999, {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_perf(1000, 1999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func},
                 {UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_perf(2000, 2999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
    expect_perf(3000, 3999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func},
                 {UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func}});
    expect_perf(4000, 4999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
    expect_empty_range(5000, SIZE_MAX);
}

UCS_TEST_F(test_proto_perf, intersect_last)
{
    /*
     * 500   1000         1999    3000  3499    3999
     *  |     |             |      |      |       |
     *  |     +- local_tl +-+      +- remote_tl --+
     *  |                                  |
     *  |                                  |
     *  +----- local_cpu ------------------+
     */
    m_perf = create();
    add_func(1000, 1999, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);
    add_func(3000, 3999, UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func);
    add_func(500, 3499, UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func);

    make_flat_perf();
    print_perf();

    expect_empty_range(0, 499);
    expect_perf(500, 999, {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
    expect_perf(1000, 1999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func},
                 {UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_perf(2000, 2999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
    expect_perf(3000, 3499,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func},
                 {UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func}});
    expect_perf(3500, 3999,
                {{UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func}});
    expect_empty_range(4000, SIZE_MAX);
}

UCS_TEST_F(test_proto_perf, intersect_middle)
{
    /*
     * 500   1000                  1999   2999
     *  |     |                      |     |
     *  +-----+------ local_tl ------+-----+
     *        |                      |
     *        |                      |
     *        +------ local_cpu -----+
     */
    m_perf = create();
    add_func(500, 2999, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);
    add_func(1000, 1999, UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func);

    make_flat_perf();
    print_perf();

    expect_empty_range(0, 499);
    expect_perf(500, 999, {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_perf(1000, 1999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func},
                 {UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_perf(2000, 2999, {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_empty_range(3000, SIZE_MAX);
}

UCS_TEST_F(test_proto_perf, agg2)
{
    /*
     * 500   1000                  1999   2999
     *  |     |                      |     |
     *  +-----+------ local_tl ------+-----+
     *        |                      |
     *        |                      |
     *        +------ local_cpu -----+
     */
    perf_ptr_t perf1 = create();
    add_func(perf1, 500, 2999, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);
    UCS_TEST_MESSAGE << perf1.get();

    perf_ptr_t perf2 = create();
    add_func(perf2, 1000, 1999, UCP_PROTO_PERF_FACTOR_LOCAL_CPU,
             local_cpu_func);
    UCS_TEST_MESSAGE << perf2.get();

    m_perf = aggregate({perf1, perf2});

    make_flat_perf();
    print_perf();

    expect_empty_range(0, 999);
    expect_perf(1000, 1999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func},
                 {UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_empty_range(2000, SIZE_MAX);
}

UCS_TEST_F(test_proto_perf, agg3)
{
    /*
     * 500   1000  1200              1999       2999
     *  |     |      |                 |          |
     *  +-----+------+---1.local_tl ---+----------+
     *        |      |                 |          |
     *        |      |                 |          |
     *        +------+--2.local_cpu1 --+          |
     *               |                            |
     *               |                            |
     *               +--3.local_cpu2 -------------+
     *
     */
    perf_ptr_t perf1 = create();
    add_func(perf1, 500, 2999, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);
    UCS_TEST_MESSAGE << perf1.get();

    perf_ptr_t perf2 = create();
    add_func(perf2, 1000, 1999, UCP_PROTO_PERF_FACTOR_LOCAL_CPU,
             local_cpu_func);
    UCS_TEST_MESSAGE << perf2.get();

    perf_ptr_t perf3 = create();
    add_func(perf3, 1200, 2999, UCP_PROTO_PERF_FACTOR_LOCAL_CPU,
             local_cpu_func);
    UCS_TEST_MESSAGE << perf3.get();

    m_perf = aggregate({perf1, perf2, perf3});

    make_flat_perf();
    print_perf();

    expect_empty_range(0, 1199);
    expect_perf(1200, 1999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU,
                  ucs_linear_func_add(local_cpu_func, local_cpu_func)},
                 {UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_empty_range(2000, SIZE_MAX);
}

UCS_TEST_F(test_proto_perf, envelope_intersect) {
    size_t overhead_per_byte = 10;
    size_t fixed_overhead    = 1000;
    auto local_tl_factor     = ucs_linear_func_make(0, overhead_per_byte);
    auto remote_tl_factor    = ucs_linear_func_make(fixed_overhead, 0);

    /*
     * Case for testing intersection of two factors:
     *           //
     * envelope //
     *         //
     * _______//
     * -------+-----------------------------
     *       /                   REMOTE_TL
     *      /
     *     / LOCAL_TL
     */
    m_perf = create();
    add_func(0, SIZE_MAX, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_factor);
    add_func(0, SIZE_MAX, UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_factor);

    make_flat_perf();
    print_perf();

    expect_perf(0, SIZE_MAX,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_factor},
                 {UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_factor}});

    // `fixed_overhead` should be divisible by `overhead_per_byte` to have
    // accurate intersection point
    ASSERT_GE(fixed_overhead, overhead_per_byte);
    ASSERT_EQ(fixed_overhead % overhead_per_byte, 0);
    auto intersection     = fixed_overhead / overhead_per_byte;
    auto *remote_tl_range = find_lb(m_envelope_flat_perf, intersection);
    auto *local_tl_range  = find_lb(m_envelope_flat_perf, intersection + 1);

    ASSERT_NE(remote_tl_range, nullptr);
    ASSERT_NE(local_tl_range, nullptr);

    ASSERT_TRUE(ucs_linear_func_is_equal(remote_tl_factor,
                                         remote_tl_range->value, 1e-9));
    ASSERT_TRUE(ucs_linear_func_is_equal(local_tl_factor,
                                         local_tl_range->value, 1e-9));
}

class test_proto_perf_random : public test_proto_perf {
protected:
    // Generate a random perf structure
    static perf_ptr_t generate_random_perf(unsigned max_segments);

private:
    // Generate a random ascending sequence of numbers, ending with SIZE_MAX
    static std::vector<size_t> generate_random_sequence(unsigned length);
};

std::vector<size_t>
test_proto_perf_random::generate_random_sequence(unsigned length)
{
    std::vector<size_t> sequence;
    size_t value = 0;
    std::generate_n(std::back_inserter(sequence), length, [&value]() {
        value += ucs::rand_range(10000) + 1;
        return value;
    });
    sequence.emplace_back(SIZE_MAX);
    return sequence;
}

test_proto_perf::perf_ptr_t
test_proto_perf_random::generate_random_perf(unsigned max_segments)
{
    perf_ptr_t perf = create();
    size_t start    = 0;
    auto points     = generate_random_sequence(max_segments);
    for (auto point : points) {
        if (ucs::rand_range(4) > 0) {
            size_t num_factors = ucs::rand_range(UCP_PROTO_PERF_FACTOR_LAST);
            perf_factors_map_t factors_map;
            for (int i = 0; i < num_factors; ++i) {
                auto factor_id   = ucs::rand_range(UCP_PROTO_PERF_FACTOR_LAST);
                /* Avoid setting 0 BW since it can cause INF estimated time */
                auto factor_func = perf_func(1.0 * ucs::rand_range(4),
                                             1000.0 * (ucs::rand_range(4) + 1));
                factors_map.emplace(factor_id, factor_func);
            }
            add_funcs(perf, start, point, factors_map);
        }
        start = point + 1;
    }
    return perf;
}

class test_proto_perf_remote : public test_proto_perf_random {
protected:
    static perf_ptr_t create_remote(perf_ptr_t perf)
    {
        ucp_proto_perf_t *remote_perf;
        ASSERT_UCS_OK(ucp_proto_perf_remote(perf.get(), &remote_perf));
        return test_proto_perf::make_perf_ptr(remote_perf);
    }
};

UCS_TEST_F(test_proto_perf_remote, compare_with_local) {
    using factor_id_t = ucp_proto_perf_factor_id_t;
    std::vector<std::pair<factor_id_t, factor_id_t>> convert_map = {
        {UCP_PROTO_PERF_FACTOR_LOCAL_CPU, UCP_PROTO_PERF_FACTOR_REMOTE_CPU},
        {UCP_PROTO_PERF_FACTOR_LOCAL_TL, UCP_PROTO_PERF_FACTOR_REMOTE_TL},
        {UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY,
         UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY},
        {UCP_PROTO_PERF_FACTOR_LATENCY, UCP_PROTO_PERF_FACTOR_LATENCY},
        {UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY,
         UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY},
        {UCP_PROTO_PERF_FACTOR_REMOTE_TL, UCP_PROTO_PERF_FACTOR_LOCAL_TL},
        {UCP_PROTO_PERF_FACTOR_REMOTE_CPU, UCP_PROTO_PERF_FACTOR_LOCAL_CPU},
    };
    m_perf                 = generate_random_perf(100);
    perf_ptr_t remote_perf = create_remote(m_perf);

    std::vector<const ucp_proto_perf_segment_t*> segments, remote_segments;
    const ucp_proto_perf_segment_t *seg, *rseg;
    size_t seg_start, seg_end;
    ucp_proto_perf_segment_foreach_range(seg, seg_start, seg_end, m_perf.get(),
                                         0, SIZE_MAX) {
        segments.emplace_back(seg);
    }
    ucp_proto_perf_segment_foreach_range(seg, seg_start, seg_end,
                                         remote_perf.get(), 0, SIZE_MAX) {
        remote_segments.emplace_back(seg);
    }

    ASSERT_EQ(segments.size(), remote_segments.size());
    for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
        rseg = remote_segments[seg_idx];
        seg  = segments[seg_idx];
        for (const auto &convert_pair : convert_map) {
            auto func  = ucp_proto_perf_segment_func(seg, convert_pair.first);
            auto rfunc = ucp_proto_perf_segment_func(rseg, convert_pair.second);
            ASSERT_TRUE(ucs_linear_func_is_equal(func, rfunc, 1e-9));
        }
    }
}

class test_proto_perf_aggregate : public test_proto_perf_random {
protected:
    // Test aggregation of random collection of perf structures, by sampling
    // the result at specific points
    void test_random_funcs(unsigned num_perfs, unsigned num_segments);

private:
    // Collect points of sampling based on perf structure segments
    static std::vector<size_t> get_sample_points(perf_ptr_t perf);

    // Calculate the expected aggregation result at a given point. If the
    // expected result is undefined, return nullptr.
    std::unique_ptr<perf_factors_map_t>
    expected_aggregate_result(const std::vector<perf_ptr_t> &perfs,
                              size_t point);
};

std::vector<size_t>
test_proto_perf_aggregate::get_sample_points(perf_ptr_t perf)
{
    std::vector<size_t> points;

    auto seg = find_lb(perf, 0);
    while (seg != NULL) {
        auto start = ucp_proto_perf_segment_start(seg);
        auto end   = ucp_proto_perf_segment_end(seg);
        points.push_back(start);
        points.push_back(end);
        points.push_back((start + end) / 2);
        if (start > 0) {
            points.push_back(start - 1);
        }
        if (end == SIZE_MAX) {
            break;
        }

        points.push_back(end + 1);
        seg = find_lb(perf, end + 1);
    }

    return points;
}

std::unique_ptr<test_proto_perf::perf_factors_map_t>
test_proto_perf_aggregate::expected_aggregate_result(
        const std::vector<perf_ptr_t> &perfs, size_t point)
{
    perf_factors_map_t agg;
    for (auto perf : perfs) {
        ucp_proto_perf_segment_t *seg = find_lb(perf, point);
        if ((seg == NULL) || (point < ucp_proto_perf_segment_start(seg))) {
            return nullptr; // Point not found in one of the perf objects
        }

        for (int factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST;
             ++factor_id) {
            auto func = ucp_proto_perf_segment_func(
                    seg, (ucp_proto_perf_factor_id_t)factor_id);
            if (!ucs_linear_func_is_zero(func, UCP_PROTO_PERF_EPSILON)) {
                if (agg.find(factor_id) == agg.end()) {
                    agg[factor_id] = func;
                } else {
                    ucs_linear_func_add_inplace(&agg[factor_id], func);
                }
            }
        }
    }

    return std::unique_ptr<perf_factors_map_t>(new perf_factors_map_t(agg));
}

void test_proto_perf_aggregate::test_random_funcs(unsigned num_perfs,
                                                  unsigned num_segments)
{
    std::vector<perf_ptr_t> perfs;
    std::generate_n(std::back_inserter(perfs), num_perfs,
                    [num_segments]() -> perf_ptr_t {
                        return generate_random_perf(num_segments);
                    });

    std::set<size_t> sample_points;
    for (auto perf : perfs) {
        auto perf_points = get_sample_points(perf);
        std::copy(perf_points.begin(), perf_points.end(),
                  std::inserter(sample_points, sample_points.begin()));
    }

    m_perf = aggregate(perfs);

    make_flat_perf();

    for (auto point : sample_points) {
        auto expected_result = expected_aggregate_result(perfs, point);
        if (expected_result) {
            expect_perf(point, point, *expected_result.get());
        } else {
            expect_empty_range(point, point);
        }
    }

    if (testing::Test::HasFailure()) {
        for (size_t i = 0; i < perfs.size(); ++i) {
            UCS_TEST_MESSAGE << "perf " << i << ": " << perfs[i].get();
        }
        UCS_TEST_MESSAGE << "result:";
        print_perf();
    }

    m_perf.reset();
}

UCS_TEST_F(test_proto_perf_aggregate, random)
{
    for (int iter = 0; (iter < (1000 / ucs::test_time_multiplier())) &&
                       !::testing::Test::HasFailure();
         ++iter) {
        test_random_funcs(10, 10);
    }
}
