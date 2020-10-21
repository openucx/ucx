/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCG_TEST_H_
#define UCG_TEST_H_

#include <common/test.h>
#include <vector>
#include <memory>
#include "ucg/builtin/plan/builtin_plan.h"
#include "ucg/builtin/ops/builtin_ops.h"
#include "ucg/base/ucg_group.h"
#include "ucg/api/ucg_plan_component.h"
#include "ucg/api/ucg.h"

#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif


class ucg_resource_factory;

/**
 * UCG test.
 */
class ucg_test : public ::testing::Test {
public:
    ucg_test();

    ~ucg_test();

    ucg_worker_h m_ucg_worker;

protected:
    void SetUp() override {
        //init_ucg_component();
    }

    void TearDown() override {
    }

    void init_ucg_component();

    ucg_collective_type_t create_allreduce_coll_type() const;

    ucg_collective_type_t create_bcast_coll_type() const;

    ucg_collective_type_t create_barrier_coll_type() const;

    ucg_collective_params_t *create_allreduce_params() const;

    ucg_collective_params_t *create_bcast_params() const;

    ucg_resource_factory *m_resource_factory;
};

struct ucg_rank_info {
    int rank;
    int nodex_idx;
    int socket_idx;
};

struct ucg_ompi_op {
    bool commutative;
};

class ucg_resource_factory {
public:
    ucg_builtin_config_t *create_config(unsigned bcast_alg, unsigned allreduce_alg, unsigned barrier_alg) const;

    ucg_group_h create_group(const ucg_rank_info my_rank_info, const std::vector<ucg_rank_info> &rank_infos,
                             ucg_worker_h ucg_worker) const;

    ucg_group_params_t *
    create_group_params(const ucg_rank_info my_rank_info, const std::vector<ucg_rank_info> &rank_infos) const;

    ucg_collective_params_t *
    create_collective_params(const ucg_collective_modifiers modifiers, const ucg_group_member_index_t root,
                             void *send_buffer, const int count, void *recv_buffer, const size_t dt_len, void *dt_ext,
                             void *op_ext) const;

    std::vector<ucg_rank_info>
    create_balanced_rank_info(const size_t nodes, const size_t ppn, const bool map_by_socket = false) const;

private:
    static int ompi_op_is_commute(void *op);
};


#endif


