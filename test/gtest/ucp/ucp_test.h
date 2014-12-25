/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCP_TEST_H_
#define UCP_TEST_H_

extern "C" {
#include <ucp/api/ucp.h>
}
#include <ucs/gtest/test.h>


/**
 * UCP test
 */
class ucp_test : public ucs::test {

protected:
    class entity {
    public:
        entity();
        ~entity();

        void connect(const entity& other);

        ucp_ep_h ep() const;

        void flush() const;

    protected:
        ucp_context_h m_ucph;
        ucp_iface_h   m_iface;
        ucp_ep_h      m_ep;
    };
};

#endif /* UCP_TEST_H_ */
