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
    enum {
        RECV_REQ_INTERNAL = DEFAULT_PARAM_VARIANT,
        RECV_REQ_EXTERNAL   /* for a receive request that was allocated by
                               the upper layer and not by ucx */
    };

    struct request {
        bool                completed;
        bool                external;
        void                *req_mem;
        ucs_status_t        status;
        ucp_tag_recv_info_t info;
    };

    struct dt_gen_state {
        size_t              count;
        int                 started;
        uint32_t            magic;
    };

    virtual void init();

    static void request_init(void *request);

    static request* request_alloc();

    static void request_release(struct request *req);

    static void request_free(struct request *req);

    static void send_callback(void *request, ucs_status_t status);

    static void recv_callback(void *request, ucs_status_t status,
                                  ucp_tag_recv_info_t *info);

    request * send_nb(const void *buffer, size_t count, ucp_datatype_t datatype,
                      ucp_tag_t tag, int ep_index = 0);

    request * send_nbr(const void *buffer, size_t count, ucp_datatype_t datatype,
                       ucp_tag_t tag, int ep_index = 0);

    void send_b(const void *buffer, size_t count, ucp_datatype_t datatype,
                ucp_tag_t tag, int buf_index = 0);

    request * send_sync_nb(const void *buffer, size_t count, ucp_datatype_t datatype,
                           ucp_tag_t tag, int buf_index = 0);

    request* recv_nb(void *buffer, size_t count, ucp_datatype_t dt,
                     ucp_tag_t tag, ucp_tag_t tag_mask, int buf_index = 0);

    request* recv_req_nb(void *buffer, size_t count, ucp_datatype_t dt,
                         ucp_tag_t tag, ucp_tag_t tag_mask, int buf_index = 0);

    request* recv_cb_nb(void *buffer, size_t count, ucp_datatype_t dt,
                        ucp_tag_t tag, ucp_tag_t tag_mask, int buf_index = 0);

    ucs_status_t recv_b(void *buffer, size_t count, ucp_datatype_t datatype,
                        ucp_tag_t tag, ucp_tag_t tag_mask,
                        ucp_tag_recv_info_t *info, int buf_index = 0);

    ucs_status_t recv_req_b(void *buffer, size_t count, ucp_datatype_t datatype,
                            ucp_tag_t tag, ucp_tag_t tag_mask,
                            ucp_tag_recv_info_t *info, int buf_index = 0);

    ucs_status_t recv_cb_b(void *buffer, size_t count, ucp_datatype_t datatype,
                           ucp_tag_t tag, ucp_tag_t tag_mask,
                           ucp_tag_recv_info_t *info, int buf_index = 0);

    void wait(request *req, int buf_index = 0);

    void wait_and_validate(request *req);

    void wait_for_unexpected_msg(ucp_worker_h worker, double sec);

    static void* dt_common_start(size_t count);

    static void* dt_common_start_pack(void *context, const void *buffer, size_t count);

    static void* dt_common_start_unpack(void *context, void *buffer, size_t count);

    template <typename T>
    static size_t dt_packed_size(void *state);

    template <typename T>
    static size_t dt_pack(void *state, size_t offset, void *dest, size_t max_length);

    template <typename T>
    static ucs_status_t dt_unpack(void *state, size_t offset, const void *src,
                                  size_t length);

    static ucs_status_t dt_err_unpack(void *state, size_t offset, const void *src,
                                      size_t length);

    static void dt_common_finish(void *state);

    virtual bool is_external_request();

    static const uint32_t MAGIC = 0xd7d7d7d7U;
    static ucp_generic_dt_ops test_dt_uint32_ops;
    static ucp_generic_dt_ops test_dt_uint32_err_ops;
    static ucp_generic_dt_ops test_dt_uint8_ops;
    static int dt_gen_start_count;
    static int dt_gen_finish_count;
    static ucp_context_attr_t ctx_attr;
private:
    int get_worker_index(int buf_index);
public:
    int    count;
};

#endif
