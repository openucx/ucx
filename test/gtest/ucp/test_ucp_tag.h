/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef TEST_UCP_TAG_H_
#define TEST_UCP_TAG_H_

#include "ucp_test.h"


class test_ucp_tag : public ucp_test {
public:
    static ucp_params_t get_ctx_params();

protected:
    struct request {
        bool                completed;
        ucs_status_t        status;
        ucp_tag_recv_info_t info;
    };

    struct dt_gen_state {
        size_t              count;
        int                 started;
        uint32_t            magic;
    };

    virtual void init();

    virtual void cleanup();

    static void request_init(void *request);

    static void request_release(struct request *req);

    static void send_callback(void *request, ucs_status_t status);

    static void recv_callback(void *request, ucs_status_t status,
                                  ucp_tag_recv_info_t *info);

    void send_b(const void *buffer, size_t count, ucp_datatype_t datatype,
                ucp_tag_t tag);

    ucs_status_t recv_b(void *buffer, size_t count, ucp_datatype_t datatype,
                        ucp_tag_t tag, ucp_tag_t tag_mask, ucp_tag_recv_info_t *info);

    void wait(request *req);

    static void* dt_start(size_t count);

    static void* dt_start_pack(void *context, const void *buffer, size_t count);

    static void* dt_start_unpack(void *context, void *buffer, size_t count);

    static size_t dt_packed_size(void *state);

    static size_t dt_pack(void *state, size_t offset, void *dest, size_t max_length);

    static ucs_status_t dt_unpack(void *state, size_t offset, const void *src,
                                  size_t length);

    static void dt_finish(void *state);

    static const uint32_t MAGIC = 0xd7d7d7d7U;
    static const ucp_datatype_t DATATYPE;
    static ucp_generic_dt_ops test_dt_ops;
    static int dt_gen_start_count;
    static int dt_gen_finish_count;

public:
    int    count;
    entity *sender, *receiver;
};

#endif
