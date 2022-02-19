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
#include <ucp/core/ucp_worker.inl>
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

        sender().connect(&receiver(), get_ep_params());
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

    static void
    discarded_cb(void *request, ucs_status_t status, void *user_data)
    {
        /* Make Coverity happy */
        ucs_assert(user_data != NULL);

        unsigned *discarded_count_p = static_cast<unsigned*>(user_data);
        (*discarded_count_p)++;
    }

    void test_worker_discard(void *ep_flush_func,
                             void *ep_pending_add_func,
                             void *ep_pending_purge_func,
                             ucs_status_t ep_flush_comp_status = UCS_OK,
                             bool wait_for_comp = true,
                             unsigned ep_count = 8,
                             unsigned wireup_ep_count = 0,
                             unsigned wireup_aux_ep_count = 0) {
        uct_iface_ops_t ops                  = {0};
        unsigned created_wireup_aux_ep_count = 0;
        unsigned total_ep_count              = ep_count + wireup_aux_ep_count;
        unsigned discarded_count             = 0;
        void *flush_req                      = NULL;
        uct_iface_t iface;
        std::vector<uct_ep_t> eps(total_ep_count);
        std::vector<uct_ep_h> wireup_eps(wireup_ep_count);
        ucs_status_t status;

        ASSERT_LE(wireup_ep_count, ep_count);
        ASSERT_LE(wireup_aux_ep_count, wireup_ep_count);

        m_ucp_ep = sender().ep();

        ops.ep_flush         = (uct_ep_flush_func_t)ep_flush_func;
        ops.ep_pending_add   = (uct_ep_pending_add_func_t)ep_pending_add_func;
        ops.ep_pending_purge = (uct_ep_pending_purge_func_t)ep_pending_purge_func;
        ops.ep_destroy       = ep_destroy_func;
        iface.ops            = ops;

        ucp_rsc_index_t rsc_index  = UCS_BITMAP_FFS(sender().ucph()->tl_bitmap);
        ucp_worker_iface_t *wiface = ucp_worker_iface(sender().worker(),
                                                      rsc_index);
        std::vector<uct_ep_h> eps_to_discard;

        for (unsigned i = 0; i < ep_count; i++) {
            uct_ep_h discard_ep;

            eps[i].iface = &iface;
            m_created_ep_count++;

            std::vector<ucp_request_t*> pending_reqs;

            if (i < wireup_ep_count) {
                status = ucp_wireup_ep_create(m_ucp_ep, &discard_ep);
                ASSERT_UCS_OK(status);

                wireup_eps.push_back(discard_ep);
                ucp_wireup_ep_set_next_ep(discard_ep, &eps[i],
                                          UCP_NULL_RESOURCE);

                ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(discard_ep);

                if (i < wireup_aux_ep_count) {
                    eps[ep_count + created_wireup_aux_ep_count].iface = &iface;

                    ucp_worker_iface_progress_ep(wiface);

                    /* coverity[escape] */
                    wireup_ep->aux_ep        =
                            &eps[ep_count + created_wireup_aux_ep_count];
                    wireup_ep->aux_rsc_index = rsc_index;

                    created_wireup_aux_ep_count++;
                    m_created_ep_count++;
                }

                EXPECT_LE(m_created_ep_count, total_ep_count);

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

            UCS_ASYNC_BLOCK(&sender().worker()->async);
            ucp_worker_iface_progress_ep(wiface);
            ucp_worker_discard_uct_ep(m_ucp_ep, discard_ep, UCT_FLUSH_FLAG_LOCAL,
                                      UCP_NULL_RESOURCE,
                                      ep_pending_purge_count_reqs_cb,
                                      &purged_reqs_count, discarded_cb,
                                      static_cast<void*>(&discarded_count));
            UCS_ASYNC_UNBLOCK(&sender().worker()->async);

            if (ep_pending_purge_func == (void*)ep_pending_purge_func_iter_reqs) {
                EXPECT_EQ(m_pending_purge_reqs_count, purged_reqs_count);
            } else {
                EXPECT_EQ(0u, purged_reqs_count);
            }
        }

        if (!wait_for_comp) {
            /* to not do flush_worker() before sender's entity destroy */
            sender().add_err(UCS_ERR_ENDPOINT_TIMEOUT);
            goto out;
        }

        flush_req = sender().flush_worker_nb(0);
        ASSERT_FALSE(flush_req == NULL);
        ASSERT_TRUE(UCS_PTR_IS_PTR(flush_req));

        do {
            progress();

            if (!m_flush_comps.empty()) {
                uct_completion_t *comp = m_flush_comps.back();

                m_flush_comps.pop_back();
                uct_invoke_completion(comp, ep_flush_comp_status);
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

        if (ep_flush_comp_status != UCS_ERR_CANCELED) {
            EXPECT_EQ(m_created_ep_count, m_destroyed_ep_count);
        }
        EXPECT_EQ(m_created_ep_count, total_ep_count);
        /* discarded_cb is called only for UCT EPs passed to
         * ucp_worker_discard_uct_ep() */
        EXPECT_EQ(ep_count, discarded_count);

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

        EXPECT_EQ(1u, m_ucp_ep->refcount);

out:
        disconnect(sender());
        sender().cleanup();
        EXPECT_EQ(m_created_ep_count, m_destroyed_ep_count);
    }

    static void ep_destroy_func(uct_ep_h ep)
    {
        for (std::vector<uct_completion_t*>::iterator iter = m_flush_comps.begin();
             iter != m_flush_comps.end(); ++iter) {
            ucp_request_t *req = ucs_container_of(*iter, ucp_request_t,
                                                  send.state.uct_comp);
            if (req->send.discard_uct_ep.uct_ep == ep) {
                /* When UCT endpoint is destroyed, all its outstanding
                 * operations are completed with status UCS_ERR_CANCELED */
                uct_invoke_completion(&req->send.state.uct_comp, UCS_ERR_CANCELED);
                m_flush_comps.erase(iter);
                EXPECT_EQ(m_ucp_ep, req->send.ep);
                EXPECT_GT(m_ucp_ep->refcount, 0u);
                break;
            }
        }

        EXPECT_GT(m_ucp_ep->refcount, 0u);

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
    static       ucp_ep_h m_ucp_ep;
    static const unsigned m_pending_purge_reqs_count;

    static std::vector<uct_completion_t*>  m_flush_comps;
    static std::vector<uct_pending_req_t*> m_pending_reqs;
    static ep_test_info_map_t              m_ep_test_info_map;
};

unsigned test_ucp_worker_discard::m_created_ep_count               = 0;
unsigned test_ucp_worker_discard::m_destroyed_ep_count             = 0;
ucp_ep_t test_ucp_worker_discard::m_fake_ep                        = {};
ucp_ep_h test_ucp_worker_discard::m_ucp_ep                         = NULL;
const unsigned test_ucp_worker_discard::m_pending_purge_reqs_count = 10;

std::vector<uct_completion_t*>              test_ucp_worker_discard::m_flush_comps;
std::vector<uct_pending_req_t*>             test_ucp_worker_discard::m_pending_reqs;
test_ucp_worker_discard::ep_test_info_map_t test_ucp_worker_discard::m_ep_test_info_map;


UCS_TEST_P(test_ucp_worker_discard, flush_ok)
{
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ucs_empty_function_do_assert      /* ep_pending_add */,
                        (void*)ucs_empty_function                /* ep_pending_purge */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_ok)
{
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ucs_empty_function_do_assert      /* ep_pending_add */,
                        (void*)ucs_empty_function                /* ep_pending_purge */,
                        UCS_OK                                   /* ep_flush_comp_status */,
                        true                                     /* wait for the completion */,
                        8                                        /* UCT EP count */,
                        6                                        /* WIREUP EP count */,
                        3                                        /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_ok_pending_purge)
{
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ep_pending_add_save_req           /* ep_pending_add */,
                        (void*)ep_pending_purge_func_iter_reqs   /* ep_pending_purge */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_ok_pending_purge_not_wait_comp)
{
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ep_pending_add_save_req           /* ep_pending_add */,
                        (void*)ep_pending_purge_func_iter_reqs   /* ep_pending_purge */,
                        UCS_OK                                   /* ep_flush_comp_status */,
                        false                                    /* don't wait for the completion */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_ok_pending_purge)
{
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ep_pending_add_save_req           /* ep_pending_add */,
                        (void*)ep_pending_purge_func_iter_reqs   /* ep_pending_purge */,
                        UCS_OK                                   /* ep_flush_comp_status */,
                        true                                     /* wait for the completion */,
                        8                                        /* UCT EP count */,
                        6                                        /* WIREUP EP count */,
                        3                                        /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_ok_pending_purge_not_wait_comp)
{
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ep_pending_add_save_req           /* ep_pending_add */,
                        (void*)ep_pending_purge_func_iter_reqs   /* ep_pending_purge */,
                        UCS_OK                                   /* ep_flush_comp_status */,
                        false                                    /* don't wait for the completion */,
                        8                                        /* UCT EP count */,
                        6                                        /* WIREUP EP count */,
                        3                                        /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_in_progress)
{
    test_worker_discard((void*)ep_flush_func_return_in_progress /* ep_flush */,
                        (void*)ucs_empty_function_do_assert     /* ep_pending_add */,
                        (void*)ucs_empty_function               /* ep_pending_purge */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_in_progress_return_canceled)
{
    test_worker_discard((void*)ep_flush_func_return_in_progress /* ep_flush */,
                        (void*)ucs_empty_function_do_assert     /* ep_pending_add */,
                        (void*)ucs_empty_function               /* ep_pending_purge */,
                        UCS_ERR_CANCELED                        /* ep_flush_comp_status */);
}


UCS_TEST_P(test_ucp_worker_discard, flush_in_progress_not_wait_comp)
{
    test_worker_discard((void*)ep_flush_func_return_in_progress /* ep_flush */,
                        (void*)ucs_empty_function_do_assert     /* ep_pending_add */,
                        (void*)ucs_empty_function               /* ep_pending_purge */,
                        UCS_OK                                  /* ep_flush_comp_status */,
                        false                                   /* don't wait for the completion */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_in_progress)
{
    test_worker_discard((void*)ep_flush_func_return_in_progress /* ep_flush */,
                        (void*)ucs_empty_function_do_assert     /* ep_pending_add */,
                        (void*)ucs_empty_function               /* ep_pending_purge */,
                        UCS_OK                                  /* ep_flush_comp_status */,
                        true                                    /* wait for the completion */,
                        8                                       /* UCT EP count */,
                        6                                       /* WIREUP EP count */,
                        3                                       /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_in_progress_return_canceled)
{
    test_worker_discard((void*)ep_flush_func_return_in_progress /* ep_flush */,
                        (void*)ucs_empty_function_do_assert     /* ep_pending_add */,
                        (void*)ucs_empty_function               /* ep_pending_purge */,
                        UCS_ERR_CANCELED                        /* ep_flush_comp_status */,
                        true                                    /* wait for the completion */,
                        8                                       /* UCT EP count */,
                        6                                       /* WIREUP EP count */,
                        3                                       /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_in_progress_not_wait_comp)
{
    test_worker_discard((void*)ep_flush_func_return_in_progress /* ep_flush */,
                        (void*)ucs_empty_function_do_assert     /* ep_pending_add */,
                        (void*)ucs_empty_function               /* ep_pending_purge */,
                        UCS_OK                                  /* ep_flush_comp_status */,
                        false                                   /* don't wait for the completion */,
                        8                                       /* UCT EP count */,
                        6                                       /* WIREUP EP count */,
                        3                                       /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_no_resource_pending_add_busy)
{
    test_worker_discard((void*)ep_flush_func_return_3_no_resource_then_ok /* ep_flush */,
                        (void*)ucs_empty_function_return_busy             /* ep_pending_add */,
                        (void*)ucs_empty_function                         /* ep_pending_purge */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_no_resource_pending_add_busy)
{
    test_worker_discard((void*)ep_flush_func_return_3_no_resource_then_ok /* ep_flush */,
                        (void*)ucs_empty_function_return_busy             /* ep_pending_add */,
                        (void*)ucs_empty_function                         /* ep_pending_purge */,
                        UCS_OK                                            /* ep_flush_comp_status */,
                        true                                              /* wait for the completion */,
                        8                                                 /* UCT EP count */,
                        6                                                 /* WIREUP EP count */,
                        3                                                 /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_no_resource_pending_add_ok_then_busy)
{
    test_worker_discard((void*)ep_flush_func_return_3_no_resource_then_ok /* ep_flush */,
                        (void*)ep_pending_add_func_return_ok_then_busy    /* ep_pending_add */,
                        (void*)ucs_empty_function                         /* ep_pending_purge */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_no_resource_pending_add_ok_then_busy_not_wait_comp)
{
    test_worker_discard((void*)ep_flush_func_return_3_no_resource_then_ok /* ep_flush */,
                        (void*)ep_pending_add_save_req                    /* ep_pending_add */,
                        (void*)ep_pending_purge_func_iter_reqs            /* ep_pending_purge */,
                        UCS_OK                                            /* ep_flush_comp_status */,
                        false                                             /* don't wait for the completion */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_no_resource_pending_add_ok_then_busy)
{
    test_worker_discard((void*)ep_flush_func_return_3_no_resource_then_ok /* ep_flush */,
                        (void*)ep_pending_add_func_return_ok_then_busy    /* ep_pending_add */,
                        (void*)ucs_empty_function                         /* ep_pending_purge */,
                        UCS_OK                                            /* ep_flush_comp_status */,
                        true                                              /* wait for the completion */,
                        8                                                 /* UCT EP count */,
                        6                                                 /* WIREUP EP count */,
                        3                                                 /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_no_resource_pending_add_ok_then_busy_not_wait_comp)
{
    test_worker_discard((void*)ep_flush_func_return_3_no_resource_then_ok /* ep_flush */,
                        (void*)ep_pending_add_save_req                    /* ep_pending_add */,
                        (void*)ep_pending_purge_func_iter_reqs            /* ep_pending_purge */,
                        UCS_OK                                            /* ep_flush_comp_status */,
                        false                                             /* don't wait for the completion */,
                        8                                                 /* UCT EP count */,
                        6                                                 /* WIREUP EP count */,
                        3                                                 /* WIREUP AUX EP count */);
}

UCS_TEST_P(test_ucp_worker_discard, flush_ok_not_wait_comp)
{
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ucs_empty_function_do_assert      /* ep_pending_add */,
                        (void*)ucs_empty_function                /* ep_pending_purge */,
                        UCS_OK                                   /* ep_flush_comp_status */,
                        false                                    /* don't wait for the completion */);
}

UCS_TEST_P(test_ucp_worker_discard, wireup_ep_flush_ok_not_wait_comp)
{
    test_worker_discard((void*)ucs_empty_function_return_success /* ep_flush */,
                        (void*)ucs_empty_function_do_assert      /* ep_pending_add */,
                        (void*)ucs_empty_function                /* ep_pending_purge */,
                        UCS_OK                                   /* ep_flush_comp_status */,
                        false                                    /* don't wait for the completion */,
                        8                                        /* UCT EP count */,
                        6                                        /* WIREUP EP count */,
                        3                                        /* WIREUP AUX EP count */);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_worker_discard, all, "all")


class test_ucp_worker_request_leak : public ucp_test {
public:
    enum {
        LEAK_CHECK,
        LEAK_IGNORE
    };

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_TAG, LEAK_CHECK,
                               "leak_check");
        add_variant_with_value(variants, UCP_FEATURE_TAG, LEAK_IGNORE,
                               "leak_ignore");
    }

    bool ignore_leak()
    {
        return get_variant_value(0) == LEAK_IGNORE;
    }

    /// @override
    virtual ucp_worker_params_t get_worker_params()
    {
        ucp_worker_params_t params = ucp_test::get_worker_params();
        if (ignore_leak()) {
            params.field_mask |= UCP_WORKER_PARAM_FIELD_FLAGS;
            params.flags      |= UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK;
        }
        return params;
    }

    /// @override
    virtual void init()
    {
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
    }

    /// @override
    virtual void cleanup()
    {
        if (ignore_leak()) {
            // Should not have warnings if leak check is off
            ucp_test::cleanup();
        } else {
            scoped_log_handler wrap_warn(wrap_warns_logger);
            ucp_test::cleanup();
            check_leak_warnings(); // Leak check is enabled - expect warnings
        }
    }

private:
    void check_leak_warnings()
    {
        EXPECT_EQ(2u, m_warnings.size());
        for (size_t i = 0; i < m_warnings.size(); ++i) {
            std::string::size_type pos = m_warnings[i].find(
                    "not returned to mpool ucp_requests");
            EXPECT_NE(std::string::npos, pos);
        }
    }
};

UCS_TEST_P(test_ucp_worker_request_leak, tag_send_recv)
{
    ucp_request_param_t param;
    param.op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    void *sreq         = ucp_tag_send_nbx(sender().ep(), NULL, 0, 0, &param);
    ASSERT_TRUE(UCS_PTR_IS_PTR(sreq));

    void *rreq = ucp_tag_recv_nbx(receiver().worker(), NULL, 0, 0, 0, &param);
    ASSERT_TRUE(UCS_PTR_IS_PTR(rreq));

    UCS_TEST_MESSAGE << "send req: " << sreq << ", recv req: " << rreq;
    while ((ucp_request_check_status(sreq) != UCS_OK) ||
           (ucp_request_check_status(rreq) != UCS_OK)) {
        progress();
    }

    // Exit the test without releasing the requests
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_worker_request_leak, all, "all")

class test_ucp_worker_thread_mode : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_TAG,
                               UCS_THREAD_MODE_SINGLE, "single");
        add_variant_with_value(variants, UCP_FEATURE_TAG,
                               UCS_THREAD_MODE_SERIALIZED, "serialized");
        add_variant_with_value(variants, UCP_FEATURE_TAG, UCS_THREAD_MODE_MULTI,
                               "multi");
    }

    /// @override
    virtual ucp_worker_params_t get_worker_params()
    {
        ucp_worker_params_t params = ucp_test::get_worker_params();

        params.field_mask |= UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        params.thread_mode = thread_mode();
        return params;
    }

protected:
    ucs_thread_mode_t thread_mode() const
    {
        return static_cast<ucs_thread_mode_t>(get_variant_value(0));
    }
};

UCS_TEST_P(test_ucp_worker_thread_mode, query)
{
    ucp_worker_attr_t worker_attr = {};

    worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE;
    ucs_status_t status    = ucp_worker_query(sender().worker(), &worker_attr);
    ASSERT_EQ(UCS_OK, status);
    EXPECT_EQ(thread_mode(), worker_attr.thread_mode);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_worker_thread_mode, all, "all")

class test_ucp_worker_address_query : public ucp_test {
public:
    test_ucp_worker_address_query()
    {
        if (get_variant_value(0)) {
            modify_config("UNIFIED_MODE", "y");
        }
    }

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_TAG, 0, "");
        add_variant_with_value(variants, UCP_FEATURE_TAG, 1, "unified");
    }
};

UCS_TEST_P(test_ucp_worker_address_query, query)
{
    ucp_worker_attr_t worker_attr = {};

    worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS;
    ucs_status_t status    = ucp_worker_query(sender().worker(), &worker_attr);
    ASSERT_EQ(UCS_OK, status);
    ASSERT_TRUE(worker_attr.address != NULL);

    ucp_worker_address_attr_t address_attr = {};
    address_attr.field_mask = UCP_WORKER_ADDRESS_ATTR_FIELD_UID;
    status                  = ucp_worker_address_query(worker_attr.address,
                                                       &address_attr);
    ASSERT_EQ(UCS_OK, status);

    EXPECT_EQ(sender().worker()->uuid, address_attr.worker_uid);
    ucp_worker_release_address(sender().worker(), worker_attr.address);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_worker_address_query)

class test_ucp_modify_uct_cfg : public test_ucp_context {
public:
    test_ucp_modify_uct_cfg() : m_seg_size((ucs::rand() & 0x3ff) + 1024) {
        ucp_config_modify(m_ucp_config, "IB_SEG_SIZE",
                          ucs::to_string(m_seg_size).c_str());
    }

    void verify_seg_size(ucp_worker_h worker) const {
        ucp_rsc_index_t tl_id;

        UCS_BITMAP_FOR_EACH_BIT(worker->context->tl_bitmap, tl_id) {
            ucp_worker_iface_t *wiface = ucp_worker_iface(worker, tl_id);

            if (wiface->attr.cap.flags & UCT_IFACE_FLAG_PUT_BCOPY) {
                EXPECT_EQ(m_seg_size, wiface->attr.cap.put.max_bcopy)
                << "tl : " << worker->context->tl_rscs[tl_id].tl_rsc.tl_name;
            }
        }
    }

private:
    const size_t m_seg_size;
};

UCS_TEST_P(test_ucp_modify_uct_cfg, verify_seg_size)
{
    entity *e = create_entity();

    verify_seg_size(e->worker());
}

/**
 * Validate below transports in which SEG_SIZE parameter affects
 * put.max_bcopy.
 */
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_modify_uct_cfg, dcx, "dc_x")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_modify_uct_cfg, rc,  "rc_v")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_modify_uct_cfg, rcx, "rc_x")
