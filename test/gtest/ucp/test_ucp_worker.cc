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
#include <ucp/core/ucp_request.h>
#include <ucp/wireup/wireup_ep.h>
#include <uct/base/uct_iface.h>
}


class test_ucp_worker_discard : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_TAG);
    }

protected:
    struct ep_test_info_t {
        std::vector<uct_pending_req_t*>    pending_reqs;
        unsigned                           flush_count;
        unsigned                           pending_add_count;

        ep_test_info_t() : flush_count(0), pending_add_count(0) {
        }
    };
    typedef std::map<uct_ep_h, ep_test_info_t> ep_test_info_map_t;

    void init() {
        ucp_test::init();
        m_created_ep_count   = 0;
        m_destroyed_ep_count = 0;
        m_fake_ep.flags      = UCP_EP_FLAG_REMOTE_CONNECTED;

        m_flush_comps.clear();
        m_pending_reqs.clear();
        m_ep_test_info_map.clear();
    }

    void add_pending_reqs(uct_ep_h uct_ep,
                          uct_pending_callback_t func,
                          std::vector<ucp_request_t*> &pending_reqs,
                          unsigned base = 0) {
        for (unsigned i = 0; i < m_pending_purge_reqs_count; i++) {
            /* use `ucs_calloc()` here, since the memory could be released
             * in the `ucp_wireup_msg_progress()` function by `ucs_free()` */
            ucp_request_t *req = static_cast<ucp_request_t*>(
                                     ucs_calloc(1, sizeof(*req),
                                                "ucp_request"));
            ASSERT_TRUE(req != NULL);

            pending_reqs.push_back(req);

            if (func == ucp_wireup_msg_progress) {               
                req->send.ep = &m_fake_ep;
            }

            req->send.uct.func = func;
            uct_ep_pending_add(uct_ep, &req->send.uct, 0);
        }
    }

    void test_worker_discard(void *ep_flush_func,
                             void *ep_pending_add_func,
                             void *ep_pending_purge_func,
                             bool wait_for_comp = true,
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

        std::vector<uct_ep_h> eps_to_discard;

        for (unsigned i = 0; i < ep_count; i++) {
            uct_ep_h discard_ep;

            eps[i].iface = &iface;
            m_created_ep_count++;

            std::vector<ucp_request_t*> pending_reqs;

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

                if (ep_pending_purge_func == (void*)ep_pending_purge_func_iter_reqs) {
                    /* add WIREUP MSGs to the WIREUP EP (it will be added to
                     * UCT EP or WIREUP AUX EP) */
                    add_pending_reqs(discard_ep,
                                     (uct_pending_callback_t)
                                     ucp_wireup_msg_progress,
                                     pending_reqs);
                }
            } else {
                discard_ep = &eps[i];
            }

            EXPECT_LE(m_created_ep_count, total_ep_count);


            if (ep_pending_purge_func == (void*)ep_pending_purge_func_iter_reqs) {
                /* add user's pending requests */
                add_pending_reqs(discard_ep,
                                 (uct_pending_callback_t)
                                 ucs_empty_function,
                                 pending_reqs);
            }

            eps_to_discard.push_back(discard_ep);
        }

        for (std::vector<uct_ep_h>::iterator iter = eps_to_discard.begin();
             iter != eps_to_discard.end(); ++iter) {
            uct_ep_h discard_ep        = *iter;
            unsigned purged_reqs_count = 0;

            ucp_worker_discard_uct_ep(sender().worker(), discard_ep,
                                      UCT_FLUSH_FLAG_LOCAL,
                                      ep_pending_purge_count_reqs_cb,
                                      &purged_reqs_count);

            if (ep_pending_purge_func == (void*)ep_pending_purge_func_iter_reqs) {
                EXPECT_EQ(m_pending_purge_reqs_count, purged_reqs_count);
            } else {
                EXPECT_EQ(0u, purged_reqs_count);
            }
        }

        if (!wait_for_comp) {
            /* destroy sender's entity here to have an access to the valid
             * pointers */
            sender().cleanup();
            return;
        }

        void *flush_req = sender().flush_worker_nb(0);

        if (ep_flush_func != (void*)ucs_empty_function_return_success) {
            /* If uct_ep_flush() returns UCS_OK from the first call, the request
             * is not scheduled on a worker progress (it completes in-place) */
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
            ucp_request_release(flush_req);
        }

        EXPECT_EQ(m_created_ep_count, m_destroyed_ep_count);
        EXPECT_EQ(m_created_ep_count, total_ep_count);

        for (unsigned i = 0; i < m_created_ep_count; i++) {
            ep_test_info_t &test_info = ep_test_info_get(&eps[i]);

            /* check EP flush counters */
            if (ep_flush_func == (void*)ep_flush_func_return_3_no_resource_then_ok) {
                EXPECT_EQ(4, test_info.flush_count);
            } else if (ep_flush_func == (void*)ep_flush_func_return_in_progress) {
                EXPECT_EQ(1, test_info.flush_count);
            }

            /* check EP pending add counters */
            if (ep_pending_add_func == (void*)ep_pending_add_func_return_ok_then_busy) {
                /* pending_add has to be called only once per EP */
                EXPECT_EQ(1, test_info.pending_add_count);
            }
        }

        EXPECT_TRUE(m_flush_comps.empty());
        EXPECT_TRUE(m_pending_reqs.empty());

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

    static ep_test_info_t& ep_test_info_get(uct_ep_h ep) {
        ep_test_info_map_t::iterator it = m_ep_test_info_map.find(ep);

        if (it == m_ep_test_info_map.end()) {
            ep_test_info_t test_info;

            m_ep_test_info_map.insert(std::make_pair(ep, test_info));
            it = m_ep_test_info_map.find(ep);
        }

        return it->second;
    }

    static unsigned
    ep_test_info_flush_inc(uct_ep_h ep) {
        ep_test_info_t &test_info = ep_test_info_get(ep);
        return ++test_info.flush_count;
    }

    static unsigned
    ep_test_info_pending_add_inc(uct_ep_h ep) {
        ep_test_info_t &test_info = ep_test_info_get(ep);
        return ++test_info.pending_add_count;
    }

    static ucs_status_t
    ep_flush_func_return_3_no_resource_then_ok(uct_ep_h ep, unsigned flags,
                                               uct_completion_t *comp) {
        unsigned flush_ep_count = ep_test_info_flush_inc(ep);
        EXPECT_LE(flush_ep_count, 4);
        return (flush_ep_count < 4) ?
               UCS_ERR_NO_RESOURCE : UCS_OK;
    }

    static ucs_status_t
    ep_flush_func_return_in_progress(uct_ep_h ep, unsigned flags,
                                     uct_completion_t *comp) {
        unsigned flush_ep_count = ep_test_info_flush_inc(ep);
        EXPECT_LE(flush_ep_count, m_created_ep_count);
        m_flush_comps.push_back(comp);
        return UCS_INPROGRESS;
    }

    static ucs_status_t
    ep_pending_add_func_return_ok_then_busy(uct_ep_h ep, uct_pending_req_t *req,
                                            unsigned flags) {
        unsigned pending_add_ep_count = ep_test_info_pending_add_inc(ep);
        EXPECT_LE(pending_add_ep_count, m_created_ep_count);

        if (pending_add_ep_count < m_created_ep_count) {
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

        ucp_request_t *req = ucs_container_of(self,
                                              ucp_request_t,
                                              send.uct);

        ASSERT_TRUE(self->func != ucp_wireup_ep_progress_pending);
        ucs_free(req);
    }

    static ucs_status_t
    ep_pending_add_save_req(uct_ep_h ep, uct_pending_req_t *req,
                            unsigned flags) {
        ep_test_info_t &test_info = ep_test_info_get(ep);
        test_info.pending_reqs.push_back(req);
        return UCS_OK;
    }

    static void
    ep_pending_purge_func_iter_reqs(uct_ep_h ep,
                                    uct_pending_purge_callback_t cb,
                                    void *arg) {
        ep_test_info_t &test_info = ep_test_info_get(ep);
        uct_pending_req_t *req;

        for (unsigned i = 0; i < m_pending_purge_reqs_count; i++) {
            std::vector<uct_pending_req_t*> &req_vec = test_info.pending_reqs;
            if (req_vec.size() == 0) {
                break;
            }

            req = req_vec.back();
            req_vec.pop_back();
            cb(req, arg);
        }
    }

protected:
    static       unsigned m_created_ep_count;
    static       unsigned m_destroyed_ep_count;
    static       ucp_ep_t m_fake_ep;
    static const unsigned m_pending_purge_reqs_count;

    static std::vector<uct_completion_t*>  m_flush_comps;
    static std::vector<uct_pending_req_t*> m_pending_reqs;
    static ep_test_info_map_t              m_ep_test_info_map;
};

unsigned test_ucp_worker_discard::m_created_ep_count               = 0;
unsigned test_ucp_worker_discard::m_destroyed_ep_count             = 0;
ucp_ep_t test_ucp_worker_discard::m_fake_ep                        = {};
const unsigned test_ucp_worker_discard::m_pending_purge_reqs_count = 10;

std::vector<uct_completion_t*>              test_ucp_worker_discard::m_flush_comps;
std::vector<uct_pending_req_t*>             test_ucp_worker_discard::m_pending_reqs;
test_ucp_worker_discard::ep_test_info_map_t test_ucp_worker_discard::m_ep_test_info_map;


UCS_TEST_P(test_ucp_worker_discard, flush_ok) {
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ucs_empty_function_do_assert      /* ep_pending_add */,
                        (void*)ucs_empty_function                /* ep_pending_purge */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_ok) {
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ucs_empty_function_do_assert      /* ep_pending_add */,
                        (void*)ucs_empty_function                /* ep_pending_purge */,
                        true                                     /* wait for the completion */,
                        8                                        /* UCT EP count */,
                        6                                        /* WIREUP EP count */,
                        3                                        /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_ok_pending_purge) {
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ep_pending_add_save_req           /* ep_pending_add */,
                        (void*)ep_pending_purge_func_iter_reqs   /* ep_pending_purge */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_ok_pending_purge) {
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ep_pending_add_save_req           /* ep_pending_add */,
                        (void*)ep_pending_purge_func_iter_reqs   /* ep_pending_purge */,
                        true                                     /* wait for the completion */,
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
                        true                                    /* wait for the completion */,
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
                        true                                              /* wait for the completion */,
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
                        true                                              /* wait for the completion */,
                        8                                                 /* UCT EP count */,
                        6                                                 /* WIREUP EP count */,
                        3                                                 /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_ok_not_wait_comp) {
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ucs_empty_function_do_assert      /* ep_pending_add */,
                        (void*)ucs_empty_function                /* ep_pending_purge */,
                        false                                    /* don't wait for the completion */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_ok_not_wait_comp) {
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ucs_empty_function_do_assert      /* ep_pending_add */,
                        (void*)ucs_empty_function                /* ep_pending_purge */,
                        false                                    /* don't wait for the completion */,
                        8                                        /* UCT EP count */,
                        6                                        /* WIREUP EP count */,
                        3                                        /* WIREUP AUX EP count */);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_worker_discard, all, "all")
