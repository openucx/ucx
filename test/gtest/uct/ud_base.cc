#include "ud_base.h"



void ud_base_test::init() 
{
    uct_test::init();

    m_e1 = uct_test::create_entity(0);
    m_entities.push_back(m_e1);

    m_e2 = uct_test::create_entity(0);
    m_entities.push_back(m_e2);
}

uct_ud_ep_t *ud_base_test::ep(entity *e) 
{
    return ucs_derived_of(e->ep(0), uct_ud_ep_t);
}

uct_ud_ep_t *ud_base_test::ep(entity *e, int i) 
{
    return ucs_derived_of(e->ep(i), uct_ud_ep_t);
}

uct_ud_iface_t *ud_base_test::iface(entity *e) 
{
    return ucs_derived_of(e->iface(), uct_ud_iface_t);
}

void ud_base_test::short_progress_loop(double delta_ms) const
{
    uct_test::short_progress_loop(delta_ms);
}

void ud_base_test::connect() 
{
    m_e1->connect(0, *m_e2, 0);
    m_e2->connect(0, *m_e1, 0);
}

void ud_base_test::cleanup() 
{
    uct_test::cleanup();
}

ucs_status_t ud_base_test::tx(entity *e) 
{
    ucs_status_t err;
    err = uct_ep_put_short(e->ep(0), &m_dummy, sizeof(m_dummy), (uint64_t)&m_dummy, 0);
    return err;
}

ucs_status_t ud_base_test::ep_flush_b(entity *e)
{
    ucs_status_t status;
    
    do {
        short_progress_loop();
        status = uct_ep_flush(e->ep(0), 0, NULL);
    } while (status == UCS_INPROGRESS || status == UCS_ERR_NO_RESOURCE);

    return status;
}

ucs_status_t ud_base_test::iface_flush_b(entity *e)
{
    ucs_status_t status;
    
    do {
        short_progress_loop();
        status = uct_iface_flush(e->iface(), 0, NULL);
    } while (status == UCS_INPROGRESS || status == UCS_ERR_NO_RESOURCE);

    return status;
}


void ud_base_test::set_tx_win(entity *e, uct_ud_psn_t size) 
{
    /* force window */
    ep(e)->tx.max_psn = ep(e)->tx.acked_psn + size;
    ep(e)->ca.cwnd = size;
}

void ud_base_test::disable_async(entity *e) 
{
    ucs_async_remove_handler(iface(e)->async.timer_id, 1);
}


