/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCX_TEST_OP_H_
#define UCX_TEST_OP_H_

#include <uct/base/uct_md.h>
#include "ucg_test.h"
#include "ucg/builtin/ops/builtin_ops.h"

class ucg_op_test : public ucg_test {
public:
    ucg_op_test();

    virtual ~ucg_op_test();

protected:
    ucg_builtin_plan_phase_t *create_phase(ucg_builtin_plan_method_type method);

    void init_phase(ucg_builtin_plan_method_type method, ucg_builtin_plan_phase_t *phase);

    void destroy_phase(ucg_builtin_plan_phase_t *phase);

    ucg_group_h create_group();

    ucg_plan_t *create_plan(unsigned phs_cnt, ucg_collective_params_t *params, ucg_group_h group);

    ucg_builtin_plan_t *create_method_plan(ucg_builtin_plan_method_type method);

    uct_md_h create_md();
};

#endif
