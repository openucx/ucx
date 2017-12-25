/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef CONTEXT_H_
#define CONTEXT_H_

#include <ucp/api/ucp.h>

#include <atomic>


/**
 * Context wrapper - allocates and releases ucp context
 */
class context {
public:
    context() : ucp_context(nullptr), ref_count(0) {}


    ucs_status_t ref_context();


    void deref_context();


    ~context();


    ucp_context_h get_ucp_context();

private:
    ucp_context_h       ucp_context;
    std::atomic<size_t> ref_count;


    ucs_status_t create_context();


    void release_context();
};


#endif /* CONTEXT_H_ */
