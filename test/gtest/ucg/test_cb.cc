/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "test_op.h"
#include "ucg/builtin/ops/builtin_cb.inl"

using namespace std;

class ucg_cb_test : public ucg_op_test {
public:
    ucg_cb_test() {
        num_procs = 2;
    }

    ~ ucg_cb_test() = default;

public:
    ucg_builtin_request_t* create_request(ucg_builtin_op_step_t *step);
};

ucg_builtin_request_t* ucg_cb_test::create_request(ucg_builtin_op_step_t *step) {
    ucg_builtin_comp_slot_t *slot = new ucg_builtin_comp_slot_t;
    slot->cb = NULL;
    slot->step_idx = 1;
    ucs_list_head_init(&slot->msg_head);

    ucg_builtin_request_t *req = &slot->req;
    req->step = step;
    ucg_request_t *comp_req = new ucg_request_t;
    comp_req->flags = 0;
    comp_req->status = UCS_OK;
    req->comp_req = comp_req;
    ucg_builtin_op_t *op = new ucg_builtin_op_t;
    op->final_cb = NULL;
    req->op = op;

    return req;
}

static void reduce_mock(void *mpi_op, char *src_buffer, char *dst_buffer, unsigned dcount, void *mpi_datatype) {
    // do nothing
}

TEST_F(ucg_cb_test, test_op_no_optimization) {
    ucg_builtin_op_t *op = new ucg_builtin_op_t;
    ucs_status_t ret = ucg_builtin_no_optimization(op);

    ASSERT_EQ(UCS_OK, ret);
}

TEST_F(ucg_cb_test, test_op_cb_init) {
    ucg_group_h group = create_group();
    ucg_collective_params_t *params = create_allreduce_params();
    ucg_plan_t *plan = create_plan(1, params, group);

    ucg_op_t *op = new ucg_op_t();

    ucs_status_t ret = ucg_builtin_op_create(plan, params, &op);
    ASSERT_EQ(UCS_OK, ret);

    op->params = *params;
    op->plan = plan;
    ucg_builtin_op_step_t *step = &((ucg_builtin_op_t *)op)->steps[0];
    int *send_buf = (int *)params->send.buf;
    int *recv_buf = (int *)params->recv.buf;

    ucg_builtin_init_reduce((ucg_builtin_op_t *)op);
    ASSERT_EQ(send_buf[0], recv_buf[0]);

    recv_buf[0] = -1;
    ucg_builtin_init_allgather_recursive((ucg_builtin_op_t *)op);
    ASSERT_EQ(send_buf[0], recv_buf[0]);

    recv_buf[0] = -1;
    plan->group_id = 0;
    ucg_builtin_init_gather((ucg_builtin_op_t *)op);
    ASSERT_EQ(send_buf[0], recv_buf[0]);

    recv_buf[0] = -1;
    step->buf_len_unit = step->buffer_length;
    ucg_builtin_init_allgather((ucg_builtin_op_t *)op);
    ASSERT_EQ(send_buf[0], recv_buf[0]);

    recv_buf[0] = -1;
    //proc_count = 2
    plan->my_index = 1;
    step->buf_len_unit = step->buffer_length / 2;
    ucg_builtin_init_alltoall((ucg_builtin_op_t *)op);
    ASSERT_EQ(send_buf[1], recv_buf[0]);

    recv_buf[0] = -1;
    step->remote_offset = 0;
    ucg_builtin_init_ring((ucg_builtin_op_t *)op);
    ASSERT_EQ(send_buf[0], recv_buf[0]);

    // do nothing
    ucg_builtin_init_dummy((ucg_builtin_op_t *)op);

    params->send.buf = MPI_IN_PLACE;
    ucg_builtin_init_reduce((ucg_builtin_op_t *)op);
    ASSERT_EQ(recv_buf[0], recv_buf[0]);
}

TEST_F(ucg_cb_test, test_op_cb_final) {
    ucg_group_h group = create_group();
    ucg_collective_params_t *params = create_allreduce_params();
    ucg_plan_t *plan = create_plan(1, params, group);

    ucg_op_t *op = new ucg_op_t();

    ucs_status_t ret = ucg_builtin_op_create(plan, params, &op);
    ASSERT_EQ(UCS_OK, ret);

    op->params = *params;
    op->plan = plan;
    plan->my_index = 1;
    ucg_builtin_op_step_t *step = &((ucg_builtin_op_t *)op)->steps[0];
    step->buf_len_unit = step->buffer_length;

    int count = 4;
    int *recv_buf = new int[count];
    for (int i = 0; i < count; i++) {
        recv_buf[i] = i;
    }
    step->recv_buffer = (int8_t *)recv_buf;

    ucg_builtin_request_t *req = new ucg_builtin_request_t;
    req->op = (ucg_builtin_op_t *)op;
    req->step = step;

    ucg_builtin_final_allgather(req);

    int half = count / 2;
    for (int i = 0; i < count; i++) {
        if (i < half) {
            ASSERT_EQ(i + half, recv_buf[i]);
        } else {
            ASSERT_EQ(i - half, recv_buf[i]);
        }
    }

    ucg_builtin_final_alltoall(req);
    for (int i = 0; i < count; i++) {
        ASSERT_EQ(i, recv_buf[i]);
    }
}

TEST_F(ucg_cb_test, test_send_cb) {
    ucg_builtin_plan_phase_t *phase = create_phase(UCG_PLAN_METHOD_SEND_TERMINAL);
    unsigned extra_flags = 0;
    unsigned base_am_id = 0;
    ucg_group_id_t group_id = 0;
    ucg_collective_params_t *params = create_allreduce_params();
    int8_t *current_data_buffer = NULL;
    ucg_builtin_op_step_t *step = new ucg_builtin_op_step_t();
    ucp_datatype_t dtype = ucp_dt_make_contig(sizeof(int));

    ucs_status_t ret = ucg_builtin_step_create(phase, dtype, dtype, extra_flags, base_am_id, group_id,
                                               params, &current_data_buffer, step);
    ASSERT_EQ(UCS_OK, ret);

    step->buf_len_unit = sizeof(int);
    step->am_header.step_idx = 0;
    step->displs_rule = UCG_BUILTIN_OP_STEP_DISPLS_RULE_BRUCK_ALLTOALL;

    ucg_builtin_request_t *req = new ucg_builtin_request_t;
    req->step = step;

    int *send_buf = (int *)step->send_buffer;
    int *recv_buf = (int *)step->recv_buffer;

    ucg_builtin_send_alltoall(req);
    ASSERT_EQ(recv_buf[1], send_buf[0]);
}

TEST_F(ucg_cb_test, test_recv_cb_recv_one) {
    ucg_builtin_plan_phase_t *phase = create_phase(UCG_PLAN_METHOD_RECV_TERMINAL);
    unsigned extra_flags = UCG_BUILTIN_OP_STEP_FLAG_LAST_STEP;
    unsigned base_am_id = 0;
    ucg_group_id_t group_id = 0;
    ucg_collective_params_t *params = create_allreduce_params();
    int8_t *current_data_buffer = NULL;
    ucg_builtin_op_step_t *step = new ucg_builtin_op_step_t();
    ucp_datatype_t dtype = ucp_dt_make_contig(sizeof(int));

    ucs_status_t ret = ucg_builtin_step_create(phase, dtype, dtype, extra_flags, base_am_id, group_id,
                                               params, &current_data_buffer, step);
    ASSERT_EQ(UCS_OK, ret);

    ucg_builtin_request_t *req = create_request(step);

    int *recv_buf = (int *)step->recv_buffer;
    uint64_t offset = 0;
    int count = 2;
    int *data = new int[count];
    for (int i = 0; i < count; i++) {
        data[i] = i;
    }
    size_t length = sizeof(int) * count;

    int ret_int = ucg_builtin_comp_recv_one_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);
    for (int i = 0; i < count; i++) {
        ASSERT_EQ(i, recv_buf[i]);
        recv_buf[i] = -1;
    }

    req->comp_req->flags = 0;
    ret_int = ucg_builtin_comp_recv_one_then_send_cb(req, offset, (void *)data, length);

    ASSERT_EQ(1, ret_int);
    for (int i = 0; i < count; i++) {
        ASSERT_EQ(i, recv_buf[i]);
    }
}

TEST_F(ucg_cb_test, test_recv_cb_recv_many) {
    ucg_builtin_plan_phase_t *phase = create_phase(UCG_PLAN_METHOD_RECV_TERMINAL);
    unsigned extra_flags = UCG_BUILTIN_OP_STEP_FLAG_LAST_STEP;
    unsigned base_am_id = 0;
    ucg_group_id_t group_id = 0;
    ucg_collective_params_t *params = create_allreduce_params();
    int8_t *current_data_buffer = NULL;
    ucg_builtin_op_step_t *step = new ucg_builtin_op_step_t();
    ucp_datatype_t dtype = ucp_dt_make_contig(sizeof(int));

    ucs_status_t ret = ucg_builtin_step_create(phase, dtype, dtype, extra_flags, base_am_id, group_id,
                                               params, &current_data_buffer, step);
    ASSERT_EQ(UCS_OK, ret);

    ucg_builtin_request_t *req = create_request(step);

    uint64_t offset = 0;
    int count = 2;
    int *data = new int[count];
    for (int i = 0; i < count; i++) {
        data[i] = i;
    }
    size_t length = sizeof(int) * count;

    req->pending = 2;
    int ret_int = ucg_builtin_comp_recv_many_cb(req, offset, (void *)data, length);
    ASSERT_EQ(0, ret_int);
    ret_int = ucg_builtin_comp_recv_many_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);

    req->comp_req->flags = 0;
    req->pending = 2;
    ret_int = ucg_builtin_comp_recv_many_then_send_cb(req, offset, (void *)data, length);
    ASSERT_EQ(0, ret_int);
    ret_int = ucg_builtin_comp_recv_many_then_send_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);

    req->comp_req->flags = 0;
    uint8_t frag_pending = 1;
    step->fragment_pending = &frag_pending;
    step->fragment_length = 1;
    step->iter_offset = UCG_BUILTIN_OFFSET_PIPELINE_PENDING;
    ret_int = ucg_builtin_comp_recv_many_then_send_pipe_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);
    ASSERT_EQ(UCG_BUILTIN_FRAG_PENDING, step->fragment_pending[offset / step->fragment_length]);

    frag_pending = 1;
    step->fragment_pending = &frag_pending;
    step->iter_offset = UCG_BUILTIN_OFFSET_PIPELINE_READY;
    ret_int = ucg_builtin_comp_recv_many_then_send_pipe_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);
}

TEST_F(ucg_cb_test, test_recv_cb_reduce_one) {
    ucg_builtin_plan_phase_t *phase = create_phase(UCG_PLAN_METHOD_RECV_TERMINAL);
    unsigned extra_flags = UCG_BUILTIN_OP_STEP_FLAG_LAST_STEP;
    unsigned base_am_id = 0;
    ucg_group_id_t group_id = 0;
    ucg_collective_params_t *params = create_allreduce_params();
    int8_t *current_data_buffer = NULL;
    ucg_builtin_op_step_t *step = new ucg_builtin_op_step_t();
    ucp_datatype_t dtype = ucp_dt_make_contig(sizeof(int));

    ucs_status_t ret = ucg_builtin_step_create(phase, dtype, dtype, extra_flags, base_am_id, group_id,
                                               params, &current_data_buffer, step);
    ASSERT_EQ(UCS_OK, ret);

    ucg_builtin_request_t *req = create_request(step);

    uint64_t offset = 0;
    int count = 2;
    int *data = new int[count];
    for (int i = 0; i < count; i++) {
        data[i] = i;
    }
    size_t length = sizeof(int) * count;

    params->recv.count = count;
    params->recv.dt_len = sizeof(int);
    req->op->super.params = *params;
    ucg_builtin_mpi_reduce_cb = reduce_mock;

    int ret_int = ucg_builtin_comp_reduce_one_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);

    req->comp_req->flags = 0;
    ret_int = ucg_builtin_comp_reduce_one_then_send_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);
}

TEST_F(ucg_cb_test, test_recv_cb_reduce_many) {
    ucg_builtin_plan_phase_t *phase = create_phase(UCG_PLAN_METHOD_RECV_TERMINAL);
    unsigned extra_flags = UCG_BUILTIN_OP_STEP_FLAG_LAST_STEP;
    unsigned base_am_id = 0;
    ucg_group_id_t group_id = 0;
    ucg_collective_params_t *params = create_allreduce_params();
    int8_t *current_data_buffer = NULL;
    ucg_builtin_op_step_t *step = new ucg_builtin_op_step_t();
    ucp_datatype_t dtype = ucp_dt_make_contig(sizeof(int));

    ucs_status_t ret = ucg_builtin_step_create(phase, dtype, dtype, extra_flags, base_am_id, group_id,
                                               params, &current_data_buffer, step);
    ASSERT_EQ(UCS_OK, ret);

    ucg_builtin_request_t *req = create_request(step);

    uint64_t offset = 0;
    int count = 2;
    int *data = new int[count];
    for (int i = 0; i < count; i++) {
        data[i] = i;
    }
    size_t length = sizeof(int) * count;

    params->recv.count = count;
    params->recv.dt_len = sizeof(int);
    req->op->super.params = *params;
    ucg_builtin_mpi_reduce_cb = reduce_mock;

    req->pending = 2;
    int ret_int = ucg_builtin_comp_reduce_many_cb(req, offset, (void *)data, length);
    ASSERT_EQ(0, ret_int);
    ret_int = ucg_builtin_comp_reduce_many_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);

    req->comp_req->flags = 0;
    req->pending = 2;
    ret_int = ucg_builtin_comp_reduce_many_then_send_cb(req, offset, (void *)data, length);
    ASSERT_EQ(0, ret_int);
    ret_int = ucg_builtin_comp_reduce_many_then_send_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);

    req->comp_req->flags = 0;
    req->pending = 2;
    uint8_t frag_pending = 1;
    step->fragment_pending = &frag_pending;
    step->fragment_length = 1;
    step->iter_offset = UCG_BUILTIN_OFFSET_PIPELINE_PENDING;
    ret_int = ucg_builtin_comp_reduce_many_then_send_pipe_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);
}

TEST_F(ucg_cb_test, test_recv_cb_reduce_full) {
    ucg_builtin_plan_phase_t *phase = create_phase(UCG_PLAN_METHOD_RECV_TERMINAL);
    unsigned extra_flags = UCG_BUILTIN_OP_STEP_FLAG_LAST_STEP;
    unsigned base_am_id = 0;
    ucg_group_id_t group_id = 0;
    ucg_collective_params_t *params = create_allreduce_params();
    int8_t *current_data_buffer = NULL;
    ucg_builtin_op_step_t *step = new ucg_builtin_op_step_t();
    ucp_datatype_t dtype = ucp_dt_make_contig(sizeof(int));

    ucs_status_t ret = ucg_builtin_step_create(phase, dtype, dtype, extra_flags, base_am_id, group_id,
                                               params, &current_data_buffer, step);
    ASSERT_EQ(UCS_OK, ret);

    ucg_builtin_request_t *req = create_request(step);

    uint64_t offset = 0;
    int count = 2;
    int *data = new int[count];
    int *cache = new int[count];
    for (int i = 0; i < count; i++) {
        data[i] = i;
        cache[i] = -1;
    }
    size_t length = sizeof(int) * count;

    phase->recv_cache_buffer = (int8_t *)cache;
    ucg_builtin_mpi_reduce_cb = reduce_mock;

    req->pending = 1;
    int ret_int = ucg_builtin_comp_reduce_full_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);
    for (int i = 0; i < count; i++) {
        ASSERT_EQ(i, cache[i]);
        cache[i] = -1;
    }

    req->comp_req->flags = 0;
    req->pending = 1;
    ret_int = ucg_builtin_comp_reduce_full_then_send_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);
    for (int i = 0; i < count; i++) {
        ASSERT_EQ(i, cache[i]);
    }
}

TEST_F(ucg_cb_test, test_recv_cb_wait) {
    ucg_builtin_plan_phase_t *phase = create_phase(UCG_PLAN_METHOD_RECV_TERMINAL);
    unsigned extra_flags = UCG_BUILTIN_OP_STEP_FLAG_LAST_STEP;
    unsigned base_am_id = 0;
    ucg_group_id_t group_id = 0;
    ucg_collective_params_t *params = create_allreduce_params();
    int8_t *current_data_buffer = NULL;
    ucg_builtin_op_step_t *step = new ucg_builtin_op_step_t();
    ucp_datatype_t dtype = ucp_dt_make_contig(sizeof(int));

    ucs_status_t ret = ucg_builtin_step_create(phase, dtype, dtype, extra_flags, base_am_id, group_id,
                                               params, &current_data_buffer, step);
    ASSERT_EQ(UCS_OK, ret);

    ucg_builtin_request_t *req = create_request(step);

    uint64_t offset = 0;
    int count = 2;
    int *data = new int[count];
    size_t length = sizeof(int) * count;

    int ret_int = ucg_builtin_comp_wait_one_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);

    req->comp_req->flags = 0;
    ret_int = ucg_builtin_comp_wait_one_then_send_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);

    req->comp_req->flags = 0;
    req->pending = 2;
    ret_int = ucg_builtin_comp_wait_many_cb(req, offset, (void *)data, length);
    ASSERT_EQ(0, ret_int);

    ret_int = ucg_builtin_comp_wait_many_then_send_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);
}

TEST_F(ucg_cb_test, test_recv_cb_last_barrier) {
    ucg_builtin_plan_phase_t *phase = create_phase(UCG_PLAN_METHOD_RECV_TERMINAL);
    unsigned extra_flags = UCG_BUILTIN_OP_STEP_FLAG_LAST_STEP;
    unsigned base_am_id = 0;
    ucg_group_id_t group_id = 0;
    ucg_collective_params_t *params = create_allreduce_params();
    int8_t *current_data_buffer = NULL;
    ucg_builtin_op_step_t *step = new ucg_builtin_op_step_t();
    ucp_datatype_t dtype = ucp_dt_make_contig(sizeof(int));

    ucs_status_t ret = ucg_builtin_step_create(phase, dtype, dtype, extra_flags, base_am_id, group_id,
                                               params, &current_data_buffer, step);
    ASSERT_EQ(UCS_OK, ret);

    ucg_builtin_request_t *req = create_request(step);

    ucg_group_h group = create_group();
    ucg_plan_t *plan = create_plan(1, params, group);
    req->op->super.plan = plan;

    uint64_t offset = 0;
    int count = 2;
    int *data = new int[count];
    size_t length = sizeof(int) * count;

    group->is_barrier_outstanding = 1;
    int ret_int = ucg_builtin_comp_last_barrier_step_one_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);

    req->comp_req->flags = 0;
    group->is_barrier_outstanding = 1;
    req->pending = 2;
    ret_int = ucg_builtin_comp_last_barrier_step_many_cb(req, offset, (void *)data, length);
    ASSERT_EQ(0, ret_int);
    ret_int = ucg_builtin_comp_last_barrier_step_many_cb(req, offset, (void *)data, length);
    ASSERT_EQ(1, ret_int);
}

TEST_F(ucg_cb_test, test_zcopy_step_check_cb) {
    ucg_builtin_plan_phase_t *phase = create_phase(UCG_PLAN_METHOD_RECV_TERMINAL);
    unsigned extra_flags = UCG_BUILTIN_OP_STEP_FLAG_LAST_STEP;
    unsigned base_am_id = 0;
    ucg_group_id_t group_id = 0;
    ucg_collective_params_t *params = create_allreduce_params();
    int8_t *current_data_buffer = NULL;
    ucg_builtin_op_step_t *step = new ucg_builtin_op_step_t();
    ucp_datatype_t dtype = ucp_dt_make_contig(sizeof(int));

    ucs_status_t ret = ucg_builtin_step_create(phase, dtype, dtype, extra_flags, base_am_id, group_id,
                                               params, &current_data_buffer, step);
    ASSERT_EQ(UCS_OK, ret);

    ucg_builtin_request_t *req = create_request(step);
    ucg_builtin_zcomp_t *zcomp = new ucg_builtin_zcomp_t;
    zcomp->req = req;
    uct_completion_t *self = &zcomp->comp;

    req->pending = 1;
    step->zcopy.num_store = 0;
    ucg_builtin_step_am_zcopy_comp_step_check_cb(self, UCS_OK);

    req->pending = 2;
    step->zcopy.num_store = 1;
    ucg_plan_t *plan = new ucg_plan_t;
    plan->planner = &ucg_builtin_component;
    req->op->super.plan = plan;
    ucg_builtin_step_am_zcopy_comp_step_check_cb(self, UCS_OK);
}

TEST_F(ucg_cb_test, test_zcopy_prep) {
    ucg_builtin_plan_phase_t *phase = create_phase(UCG_PLAN_METHOD_RECV_TERMINAL);
    unsigned extra_flags = UCG_BUILTIN_OP_STEP_FLAG_LAST_STEP;
    unsigned base_am_id = 0;
    ucg_group_id_t group_id = 0;
    ucg_collective_params_t *params = create_allreduce_params();
    int8_t *current_data_buffer = NULL;
    ucg_builtin_op_step_t *step = new ucg_builtin_op_step_t();
    ucp_datatype_t dtype = ucp_dt_make_contig(sizeof(int));

    ucs_status_t ret = ucg_builtin_step_create(phase, dtype, dtype, extra_flags, base_am_id, group_id,
                                               params, &current_data_buffer, step);
    ASSERT_EQ(UCS_OK, ret);

    uct_md_h md = create_md();
    step->uct_md = md;

    ret = ucg_builtin_step_zcopy_prep(step);
    ASSERT_EQ(UCS_OK, ret);
}

TEST_F(ucg_cb_test, test_bcopy_to_zcopy) {
    ucg_builtin_plan_phase_t *phase = create_phase(UCG_PLAN_METHOD_RECV_TERMINAL);
    unsigned extra_flags = UCG_BUILTIN_OP_STEP_FLAG_LAST_STEP | UCG_BUILTIN_OP_STEP_FLAG_SEND_AM_BCOPY;
    unsigned base_am_id = 0;
    ucg_group_id_t group_id = 0;
    ucg_collective_params_t *params = create_allreduce_params();
    int8_t *current_data_buffer = NULL;
    ucg_builtin_op_step_t *step = new ucg_builtin_op_step_t();
    ucp_datatype_t dtype = ucp_dt_make_contig(sizeof(int));

    ucs_status_t ret = ucg_builtin_step_create(phase, dtype, dtype, extra_flags, base_am_id, group_id,
                                               params, &current_data_buffer, step);
    ASSERT_EQ(UCS_OK, ret);

    uct_md_h md = create_md();
    step->uct_md = md;
    step->recv_cb = ucg_builtin_comp_reduce_one_cb;

    ucg_builtin_op_t *op = new ucg_builtin_op_t + sizeof(ucg_builtin_op_step_t);
    op->steps[0] = *step;

    ret = ucg_builtin_optimize_bcopy_to_zcopy(op);
    ASSERT_EQ(UCS_OK, ret);
}

/**
 * Test: ucg_builtin_step_select_callbacks
 */
TEST_F(ucg_cb_test, test_step_callback_select_termonal) {
    ucg_builtin_plan_method_type method[] = {UCG_PLAN_METHOD_SEND_TERMINAL, UCG_PLAN_METHOD_RECV_TERMINAL,
                                             UCG_PLAN_METHOD_SCATTER_TERMINAL};

    int len = sizeof(method) / sizeof(method[0]);
    for (int i = 0; i < len; i++) {
        ucg_builtin_plan_phase_t *phase = create_phase(method[i]);
        ucg_builtin_comp_recv_cb_t *recv_cb = new ucg_builtin_comp_recv_cb_t();
        int nonzero_length = 0;
        int flags = 0;

        ucs_status_t ret = ucg_builtin_step_select_callbacks(phase, recv_cb, nonzero_length, flags);
        ASSERT_EQ(UCS_OK, ret);

        flags |= UCG_BUILTIN_OP_STEP_FLAG_SINGLE_ENDPOINT;
        ret = ucg_builtin_step_select_callbacks(phase, recv_cb, nonzero_length, flags);
        ASSERT_EQ(UCS_OK, ret);
    }
}

TEST_F(ucg_cb_test, test_step_callback_select_waypoint_fanout) {
    ucg_builtin_plan_method_type method[] = {UCG_PLAN_METHOD_BCAST_WAYPOINT, UCG_PLAN_METHOD_SCATTER_WAYPOINT,
                                             UCG_PLAN_METHOD_GATHER_WAYPOINT};

    int len = sizeof(method) / sizeof(method[0]);
    for (int i = 0; i < len; i++) {
        ucg_builtin_plan_phase_t *phase = create_phase(method[i]);
        ucg_builtin_comp_recv_cb_t *recv_cb = new ucg_builtin_comp_recv_cb_t();
        int nonzero_length = 0;
        int flags = 0;

        ucs_status_t ret = ucg_builtin_step_select_callbacks(phase, recv_cb, nonzero_length, flags);
        ASSERT_EQ(UCS_OK, ret);

        nonzero_length = 1;
        ret = ucg_builtin_step_select_callbacks(phase, recv_cb, nonzero_length, flags);
        ASSERT_EQ(UCS_OK, ret);

        flags |= UCG_BUILTIN_OP_STEP_FLAG_FRAGMENTED;
        ret = ucg_builtin_step_select_callbacks(phase, recv_cb, nonzero_length, flags);
        ASSERT_EQ(UCS_OK, ret);
    }
}

TEST_F(ucg_cb_test, test_step_callback_select_reduce) {
    ucg_builtin_plan_method_type method[] = {UCG_PLAN_METHOD_REDUCE_TERMINAL, UCG_PLAN_METHOD_REDUCE_RECURSIVE,
                                             UCG_PLAN_METHOD_REDUCE_WAYPOINT};

    int len = sizeof(method) / sizeof(method[0]);
    for (int i = 0; i < len; i++) {
        ucg_builtin_plan_phase_t *phase = create_phase(method[i]);
        ucg_builtin_comp_recv_cb_t *recv_cb = new ucg_builtin_comp_recv_cb_t();

        for (int nonzero_length = 0; nonzero_length < 2; nonzero_length++) {
            int flags = 0;
            ucs_status_t ret = ucg_builtin_step_select_callbacks(phase, recv_cb, nonzero_length, flags);
            ASSERT_EQ(UCS_OK, ret);

            flags |= UCG_BUILTIN_OP_STEP_FLAG_SINGLE_ENDPOINT;
            ret = ucg_builtin_step_select_callbacks(phase, recv_cb, nonzero_length, flags);
            ASSERT_EQ(UCS_OK, ret);
        }
    }
}

TEST_F(ucg_cb_test, test_step_callback_select_nonzero) {
    ucg_builtin_plan_method_type method[] = {UCG_PLAN_METHOD_ALLGATHER_RECURSIVE,
                                             UCG_PLAN_METHOD_REDUCE_SCATTER_RING};

    int len = sizeof(method) / sizeof(method[0]);
    for (int i = 0; i < len; i++) {
        ucg_builtin_plan_phase_t *phase = create_phase(method[i]);
        ucg_builtin_comp_recv_cb_t *recv_cb = new ucg_builtin_comp_recv_cb_t();

        for (int nonzero_length = 0; nonzero_length < 2; nonzero_length++) {
            int flags = 0;
            ucs_status_t ret = ucg_builtin_step_select_callbacks(phase, recv_cb, nonzero_length, flags);
            ASSERT_EQ(UCS_OK, ret);
        }
    }
}

TEST_F(ucg_cb_test, test_step_callback_select_barrier) {
    ucg_builtin_plan_phase_t *phase = create_phase(UCG_PLAN_METHOD_ALLGATHER_RING);
    ucg_builtin_comp_recv_cb_t *recv_cb = new ucg_builtin_comp_recv_cb_t();
    int nonzero_length = 0;
    int flags = UCG_BUILTIN_OP_STEP_FLAG_LAST_STEP;

    ucs_status_t ret = ucg_builtin_step_select_callbacks(phase, recv_cb, nonzero_length, flags);
    ASSERT_EQ(UCS_OK, ret);

    flags |= UCG_BUILTIN_OP_STEP_FLAG_SINGLE_ENDPOINT;
    ret = ucg_builtin_step_select_callbacks(phase, recv_cb, nonzero_length, flags);
    ASSERT_EQ(UCS_OK, ret);
}

/**
 * Test: ucg_builtin_op_select_callback
 */
TEST_F(ucg_cb_test, test_op_collback_select) {
    ucg_builtin_plan_method_type method[] = {UCG_PLAN_METHOD_SEND_TERMINAL, UCG_PLAN_METHOD_RECV_TERMINAL,
                                             UCG_PLAN_METHOD_BCAST_WAYPOINT, UCG_PLAN_METHOD_GATHER_WAYPOINT,
                                             UCG_PLAN_METHOD_SCATTER_TERMINAL, UCG_PLAN_METHOD_SCATTER_WAYPOINT,
                                             UCG_PLAN_METHOD_REDUCE_TERMINAL, UCG_PLAN_METHOD_REDUCE_WAYPOINT,
                                             UCG_PLAN_METHOD_REDUCE_RECURSIVE, UCG_PLAN_METHOD_NEIGHBOR,
                                             UCG_PLAN_METHOD_ALLGATHER_BRUCK, UCG_PLAN_METHOD_ALLGATHER_RECURSIVE,
                                             UCG_PLAN_METHOD_ALLTOALL_BRUCK, UCG_PLAN_METHOD_REDUCE_SCATTER_RING,
                                             UCG_PLAN_METHOD_ALLGATHER_RING};

    int len = sizeof(method) / sizeof(method[0]);
    for (int i = 0; i < len; i++) {
        ucg_builtin_plan_t *plan = create_method_plan(method[i]);
        ucg_builtin_op_init_cb_t *init_cb = new ucg_builtin_op_init_cb_t;
        ucg_builtin_op_final_cb_t *final_cb = new ucg_builtin_op_final_cb_t;
        ucp_datatype_t dtype = ucp_dt_make_contig(sizeof(int));

        ucs_status_t ret = ucg_builtin_op_select_callback(plan, dtype, dtype, init_cb, final_cb);

        ASSERT_EQ(UCS_OK, ret);
    }
}

/**
 * Test: ucg_builtin_op_consider_optimization
 */
TEST_F(ucg_cb_test, test_op_consider_optimization) {
    ucg_group_h group = create_group();
    ucg_collective_params_t *params = create_bcast_params();
    ucg_plan_t *plan = create_plan(1, params, group);

    ucg_op_t *op = new ucg_op_t();

    ucs_status_t ret = ucg_builtin_op_create(plan, params, &op);
    ASSERT_EQ(UCS_OK, ret);

    ret = ucg_builtin_op_consider_optimization((ucg_builtin_op_t*)op,
                                               (ucg_builtin_config_t*)plan->planner->plan_config);
    ASSERT_EQ(UCS_OK, ret);
}
