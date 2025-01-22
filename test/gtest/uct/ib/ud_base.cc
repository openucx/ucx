/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ud_base.h"



void ud_base_test::init()
{
    uct_test::init();

    m_e1 = uct_test::create_entity(0, get_err_handler());
    m_entities.push_back(m_e1);

    check_skip_test();

    m_e2 = uct_test::create_entity(0, get_err_handler());
    m_entities.push_back(m_e2);
}

uct_error_handler_t ud_base_test::get_err_handler() const
{
    return NULL;
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

void ud_base_test::short_progress_loop(double delta_ms, entity *e) const
{
    uct_test::short_progress_loop(delta_ms, e);
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
    iface(e)->async.disable = 1;
}
