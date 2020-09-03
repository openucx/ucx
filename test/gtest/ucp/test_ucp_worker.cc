/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"
#include <uct/api/uct.h>
#include <uct/api/tl.h>

extern "C" {
#include <ucp/core/ucp_worker.h>
#include <ucp/wireup/wireup_ep.h>
#include <uct/base/uct_iface.h>
}


class test_ucp_worker_discard : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_TAG;
        return params;
    }

protected:
    void init() {
        ucp_test::init();
        m_created_ep_count     = 0;
        m_destroyed_ep_count   = 0;
        m_flush_ep_count       = 0;
        m_pending_add_ep_count = 0;

        m_flush_comps.clear();
        m_pending_reqs.clear();
    }

    void add_pending_reqs(uct_ep_h uct_ep,
                          uct_pending_callback_t func,
                          std::vector<uct_pending_req_t> &pending_reqs,
                          unsigned base = 0) {
        for (unsigned i = 0; i < m_pending_purge_reqs_count; i++) {
            pending_reqs[base + i].func = func;
            uct_ep_pending_add(uct_ep, &pending_reqs[base + i], 0);
        }
    }

    void test_worker_discard(void *ep_flush_func,
                             void *ep_pending_add_func,
                             void *ep_pending_purge_func,
                             unsigned ep_count = 8,
                             unsigned wireup_ep_count = 0,
                             unsigned wireup_aux_ep_count = 0) {
        uct_iface_ops_t ops                  = {0};
        unsigned created_wireup_aux_ep_count = 0;
        unsigned total_ep_count              = ep_count + wireup_aux_ep_count;
        uct_iface_t iface;
        std::vector<uct_ep_t> eps(total_ep_count);
        std::vector<uct_ep_h> wireup_eps(wireup_ep_count);
        ucp_ep_t ucp_ep;
        ucs_status_t status;

        ASSERT_LE(wireup_ep_count, ep_count);
        ASSERT_LE(wireup_aux_ep_count, wireup_ep_count);

        ucp_ep.worker = sender().worker();

        ops.ep_flush         = (uct_ep_flush_func_t)ep_flush_func;
        ops.ep_pending_add   = (uct_ep_pending_add_func_t)ep_pending_add_func;
        ops.ep_pending_purge = (uct_ep_pending_purge_func_t)ep_pending_purge_func;
        ops.ep_destroy       = ep_destroy_func;
        iface.ops            = ops;

        for (unsigned i = 0; i < ep_count; i++) {
            uct_ep_h discard_ep;

            eps[i].iface = &iface;
            m_created_ep_count++;

            unsigned expected_pending_purge_reqs_count = 0;
            unsigned added_pending_purge_reqs_count    = 0;

            if (ep_pending_purge_func == ep_pending_purge_func_iter) {
                /* expected purging count is the number of pending
                 * requests in the WIREUP EP (if used) and in the
                 * WIREUP AUX EP (if used) or UCT EP (if no WIREUP EP
                 * or no WIREUP AUX EP) */
                expected_pending_purge_reqs_count +=
                    m_pending_purge_reqs_count;
                
                if (i < wireup_ep_count) {
                    expected_pending_purge_reqs_count +=
                        m_pending_purge_reqs_count;
                }
            }

            std::vector<uct_pending_req_t>
                pending_reqs(expected_pending_purge_reqs_count);

            if (i < wireup_ep_count) {
                status = ucp_wireup_ep_create(&ucp_ep, &discard_ep);
                ASSERT_UCS_OK(status);

                wireup_eps.push_back(discard_ep);
                ucp_wireup_ep_set_next_ep(discard_ep, &eps[i]);

                ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(discard_ep);

                if (i < wireup_aux_ep_count) {
                    eps[ep_count + created_wireup_aux_ep_count].iface = &iface;

                    /* coverity[escape] */
                    wireup_ep->aux_ep = &eps[ep_count +
                                             created_wireup_aux_ep_count];

                    created_wireup_aux_ep_count++;
                    m_created_ep_count++;
                }

                if (expected_pending_purge_reqs_count > 0) {
                    /* add WIREUP MSGs to the WIREUP EP (it will be added to
                     * UCT EP or WIREUP AUX EP) */
                    add_pending_reqs(discard_ep,
                                     (uct_pending_callback_t)
                                     ucp_wireup_msg_progress,
                                     pending_reqs);
                    added_pending_purge_reqs_count +=
                        m_pending_purge_reqs_count;
                }
            } else {
                discard_ep = &eps[i];
            }

            EXPECT_LE(m_created_ep_count, total_ep_count);


            if (expected_pending_purge_reqs_count > 0) {
                /* add user's pending requests */
                add_pending_reqs(discard_ep,
                                 (uct_pending_callback_t)
                                 ucs_empty_function,
                                 pending_reqs,
                                 added_pending_purge_reqs_count);
                added_pending_purge_reqs_count +=
                    m_pending_purge_reqs_count;
            }

            EXPECT_EQ(expected_pending_purge_reqs_count,
                      added_pending_purge_reqs_count);

            unsigned purged_reqs_count = 0;
            ucp_worker_discard_uct_ep(sender().worker(), discard_ep,
                                      UCT_FLUSH_FLAG_LOCAL,
                                      ep_pending_purge_count_reqs_cb,
                                      &purged_reqs_count);
            EXPECT_EQ(expected_pending_purge_reqs_count, purged_reqs_count);
        }

        void *flush_req = sender().flush_worker_nb(0);

        ASSERT_FALSE(flush_req == NULL);
        ASSERT_TRUE(UCS_PTR_IS_PTR(flush_req));

        do {
            progress();

            if (!m_flush_comps.empty()) {
                uct_completion_t *comp = m_flush_comps.back();

                m_flush_comps.pop_back();
                uct_invoke_completion(comp, UCS_OK);
            }

            if (!m_pending_reqs.empty()) {
                uct_pending_req_t *req = m_pending_reqs.back();

                status = req->func(req);
                if (status == UCS_OK) {
                    m_pending_reqs.pop_back();
                } else {
                    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);
                }
            }
        } while (ucp_request_check_status(flush_req) == UCS_INPROGRESS);

        EXPECT_UCS_OK(ucp_request_check_status(flush_req));
        EXPECT_EQ(m_created_ep_count, m_destroyed_ep_count);
        EXPECT_EQ(m_created_ep_count, total_ep_count);
        EXPECT_TRUE(m_flush_comps.empty());
        EXPECT_TRUE(m_pending_reqs.empty());

        ucp_request_release(flush_req);

        /* check that uct_ep_destroy() was called for the all EPs that
         * were created in the test */
        for (unsigned i = 0; i < created_wireup_aux_ep_count; i++) {
            EXPECT_EQ(NULL, eps[i].iface);
        }
    }

    static void ep_destroy_func(uct_ep_h ep) {
        ep->iface = NULL;
        m_destroyed_ep_count++;
    }

    static ucs_status_t
    ep_flush_func_return_3_no_resource_then_ok(uct_ep_h ep, unsigned flags,
                                               uct_completion_t *comp) {
        EXPECT_LT(m_flush_ep_count, 4 * m_created_ep_count);
        return (++m_flush_ep_count < 3 * m_created_ep_count) ?
               UCS_ERR_NO_RESOURCE : UCS_OK;
    }

    static ucs_status_t
    ep_flush_func_return_in_progress(uct_ep_h ep, unsigned flags,
                                     uct_completion_t *comp) {
        EXPECT_LT(m_flush_ep_count, m_created_ep_count);
        m_flush_comps.push_back(comp);
        return UCS_INPROGRESS;
    }

    static ucs_status_t
    ep_pending_add_func_return_ok_then_busy(uct_ep_h ep, uct_pending_req_t *req,
                                            unsigned flags) {
        EXPECT_LT(m_pending_add_ep_count, 3 * m_created_ep_count);

        if (++m_pending_add_ep_count < m_created_ep_count) {
            m_pending_reqs.push_back(req);
            return UCS_OK;
        }

        return UCS_ERR_BUSY;
    }

    static void
    ep_pending_purge_count_reqs_cb(uct_pending_req_t *self,
                                   void *arg) {
        unsigned *count = (unsigned*)arg;
        (*count)++;
    }

    static ucs_status_t
    ep_pending_add_and_inc_count(uct_ep_h ep, uct_pending_req_t *req,
                                 unsigned flags) {
        std::map<uct_ep_h, unsigned>::iterator it =
            m_pending_reqs_count.find(ep);
        if (it == m_pending_reqs_count.end()) {
            m_pending_reqs_count.insert(std::make_pair(ep, 1));
        } else {
            unsigned *count = &it->second;
            (*count)++;
        }
        return UCS_OK;
    }

    static void
    ep_pending_purge_func_iter(uct_ep_h ep,
                               uct_pending_purge_callback_t cb,
                               void *arg) {
        uct_pending_req_t req = {};
        for (unsigned i = 0; i < m_pending_purge_reqs_count; i++) {
            std::map<uct_ep_h, unsigned>::iterator it =
                m_pending_reqs_count.find(ep);
            ASSERT_NE(it, m_pending_reqs_count.end());

            unsigned *count = &it->second;
            if (*count == 0) {
                break;
            }

            (*count)--;
            cb(&req, arg);
        }
    }

protected:
    static       unsigned m_created_ep_count;
    static       unsigned m_destroyed_ep_count;
    static       unsigned m_flush_ep_count;
    static       unsigned m_pending_add_ep_count;
    static const unsigned m_pending_purge_reqs_count;

    static std::vector<uct_completion_t*> m_flush_comps;
    static std::vector<uct_pending_req_t*> m_pending_reqs;
    static std::map<uct_ep_h, unsigned> m_pending_reqs_count;
};

unsigned test_ucp_worker_discard::m_created_ep_count               = 0;
unsigned test_ucp_worker_discard::m_destroyed_ep_count             = 0;
unsigned test_ucp_worker_discard::m_flush_ep_count                 = 0;
unsigned test_ucp_worker_discard::m_pending_add_ep_count           = 0;
const unsigned test_ucp_worker_discard::m_pending_purge_reqs_count = 10;

std::vector<uct_completion_t*> test_ucp_worker_discard::m_flush_comps;
std::vector<uct_pending_req_t*> test_ucp_worker_discard::m_pending_reqs;
std::map<uct_ep_h, unsigned> test_ucp_worker_discard::m_pending_reqs_count;


UCS_TEST_P(test_ucp_worker_discard, flush_ok) {
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ucs_empty_function_do_assert      /* ep_pending_add */,
                        (void*)ucs_empty_function                /* ep_pending_purge */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_ok) {
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ucs_empty_function_do_assert      /* ep_pending_add */,
                        (void*)ucs_empty_function                /* ep_pending_purge */,
                        8                                        /* UCT EP count */,
                        6                                        /* WIREUP EP count */,
                        3                                        /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_ok_pending_purge) {
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ep_pending_add_and_inc_count      /* ep_pending_add */,
                        (void*)ep_pending_purge_func_iter        /* ep_pending_purge */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_ok_pending_purge) {
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ep_pending_add_and_inc_count      /* ep_pending_add */,
                        (void*)ep_pending_purge_func_iter        /* ep_pending_purge */,
                        8                                        /* UCT EP count */,
                        6                                        /* WIREUP EP count */,
                        3                                        /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_in_progress) {
    test_worker_discard((void*)ep_flush_func_return_in_progress /* ep_flush */,
                        (void*)ucs_empty_function_do_assert     /* ep_pending_add */,
                        (void*)ucs_empty_function               /* ep_pending_purge */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_in_progress) {
    test_worker_discard((void*)ep_flush_func_return_in_progress /* ep_flush */,
                        (void*)ucs_empty_function_do_assert     /* ep_pending_add */,
                        (void*)ucs_empty_function               /* ep_pending_purge */,
                        8                                       /* UCT EP count */,
                        6                                       /* WIREUP EP count */,
                        3                                       /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_no_resource_pending_add_busy) {
    test_worker_discard((void*)ep_flush_func_return_3_no_resource_then_ok /* ep_flush */,
                        (void*)ucs_empty_function_return_busy             /* ep_pending_add */,
                        (void*)ucs_empty_function                         /* ep_pending_purge */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_no_resource_pending_add_busy) {
    test_worker_discard((void*)ep_flush_func_return_3_no_resource_then_ok /* ep_flush */,
                        (void*)ucs_empty_function_return_busy             /* ep_pending_add */,
                        (void*)ucs_empty_function                         /* ep_pending_purge */,
                        8                                                 /* UCT EP count */,
                        6                                                 /* WIREUP EP count */,
                        3                                                 /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_no_resource_pending_add_ok_then_busy) {
    test_worker_discard((void*)ep_flush_func_return_3_no_resource_then_ok /* ep_flush */,
                        (void*)ep_pending_add_func_return_ok_then_busy    /* ep_pending_add */,
                        (void*)ucs_empty_function                         /* ep_pending_purge */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_no_resource_pending_add_ok_then_busy) {
    test_worker_discard((void*)ep_flush_func_return_3_no_resource_then_ok /* ep_flush */,
                        (void*)ep_pending_add_func_return_ok_then_busy    /* ep_pending_add */,
                        (void*)ucs_empty_function                         /* ep_pending_purge */,
                        8                                                 /* UCT EP count */,
                        6                                                 /* WIREUP EP count */,
                        3                                                 /* WIREUP AUX EP count */);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_worker_discard, all, "all")
