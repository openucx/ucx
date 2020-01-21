/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
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

    enum send_type_t {
        SEND_NB,
        SEND_NBR,
        SEND_B,
        SEND_SYNC_NB
    };

    enum recv_type_t {
        RECV_NB,
        RECV_NBR,
        RECV_B,
        RECV_BR
    };

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

    virtual void init();

    void enable_tag_mp_offload();

    static void request_init(void *request);

    static request* request_alloc();

    static void request_release(struct request *req);

    static void request_free(struct request *req);

    static void send_callback(void *request, ucs_status_t status);

    static void recv_callback(void *request, ucs_status_t status,
                                  ucp_tag_recv_info_t *info);

    request* send(entity &sender, send_type_t type, const void *buffer,
                  size_t count, ucp_datatype_t datatype, ucp_tag_t tag,
                  int ep_index = 0);

    request* send_nb(const void *buffer, size_t count, ucp_datatype_t datatype,
                     ucp_tag_t tag, int ep_index = 0);

    request* send_nbr(const void *buffer, size_t count, ucp_datatype_t datatype,
                      ucp_tag_t tag, int ep_index = 0);

    void send_b(const void *buffer, size_t count, ucp_datatype_t datatype,
                ucp_tag_t tag, int buf_index = 0);

    request* send_sync_nb(const void *buffer, size_t count, ucp_datatype_t datatype,
                          ucp_tag_t tag, int buf_index = 0);

    request* recv(entity &receiver, recv_type_t type, void *buffer,
                  size_t count, ucp_datatype_t dt, ucp_tag_t tag,
                  ucp_tag_t tag_mask, ucp_tag_recv_info_t *info,
                  int buf_index = 0);

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

    void check_offload_support(bool offload_required);

    virtual bool is_external_request();

    static ucp_context_attr_t ctx_attr;
    ucs::ptr_vector<ucs::scoped_setenv> m_env;

private:
    int get_worker_index(int buf_index);

public:
    int    count;
};

#endif
