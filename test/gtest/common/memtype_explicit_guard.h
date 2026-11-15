/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef MEMTYPE_EXPLICIT_GUARD_H_
#define MEMTYPE_EXPLICIT_GUARD_H_

extern "C" {
#include <ucm/api/ucm.h>
#include <ucs/memory/memtype_cache.h>
}


/* Disables UCM events and resets the global memory type cache, and restores
 * both on destruction, also when the test fails or is skipped in init() */
class memtype_explicit_guard {
public:
    memtype_explicit_guard() : m_enable_events(0), m_active(false) {
    }

    ~memtype_explicit_guard() {
        if (m_active) {
            reset_cache();
            ucm_global_opts.enable_events = m_enable_events;
        }
    }

    void enable() {
        m_enable_events               = ucm_global_opts.enable_events;
        m_active                      = true;
        ucm_global_opts.enable_events = 0;
        reset_cache();
    }

    static void reset_cache() {
        ucs_memtype_cache_cleanup();
        ucs_memtype_cache_global_init();
    }

private:
    int  m_enable_events;
    bool m_active;
};

#endif
