/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_amo.h"

class uct_atomic_key_reg_rdma_mem_type : public uct_amo_test {
protected:
    void init() override
    {
        modify_config("IB_DM_COUNT", "0", SKIP_IF_NOT_EXIST);
        uct_amo_test::init();
    }

    bool check_rdma_memory()
    {
        FOR_EACH_ENTITY(iter) {
            if (!((*iter)->md_attr().alloc_mem_types &
                  UCS_BIT(UCS_MEMORY_TYPE_RDMA))) {
                return false;
            }
        }
        return true;
    }
};

UCS_TEST_SKIP_COND_P(uct_atomic_key_reg_rdma_mem_type, fadd64,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), FOP64) ||
                     !check_rdma_memory())
{
    mapped_buffer recvbuf(sizeof(uint64_t), receiver(), 0UL,
                          UCS_MEMORY_TYPE_RDMA);
    uint64_t add = rand64();

    run_workers(static_cast<send_func_t>(
                        &uct_amo_test::atomic_fop<uint64_t, UCT_ATOMIC_OP_ADD>),
                recvbuf, std::vector<uint64_t>(num_senders(), add), false);
    wait_for_remote();
}

UCT_INSTANTIATE_RC_DC_TEST_CASE(uct_atomic_key_reg_rdma_mem_type);
