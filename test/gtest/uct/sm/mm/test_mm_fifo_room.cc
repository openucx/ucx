/**
* Copyright (C) Advanced Micro Devices, Inc. 2025. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include <common/test.h>
#include <cstdint>
#include "uct/sm/mm/base/mm_iface.h"
#include "uct/sm/mm/base/mm_ep.h"

namespace {

static constexpr uint64_t EA  = UCT_MM_IFACE_FIFO_HEAD_EVENT_ARMED;

/* Practical upper bound for head/tail in real deployments: 2^62.
 * Rationale: head advances roughly once per nanosecond in the worst case.
 * The signed-63 midpoint is 2^62 (~4.61e18). At 1 tick/ns, reaching 2^62
 * takes ~146 years. Therefore tests cap counters at <= 2^62 to reflect
 * realistic long-running processes while still exercising wrap-related logic.
 */
static constexpr uint64_t LIM       = 1ull << 62;
/* 2^31 constant to guard against regressions to 32-bit comparisons */
static constexpr uint64_t POW2_31   = 1ull << 31;

struct case_item {
    const char *name;
    uint64_t head;
    uint64_t tail;
    unsigned fifo;
    bool expect;
};

static const case_item k_cases[] = {
    /* head < tail */
    {"lt:<fifo",        512,                     600,           256, true},
    {"lt:==fifo",       512,                     768,           256, true},
    {"lt:>fifo",        512,                     900,           256, true},
    {"EA lt:<fifo",     EA | 512,                600,           256, true},
    {"EA lt:==fifo",    EA | 512,                768,           256, true},
    {"EA lt:>fifo",     EA | 512,                900,           256, true},

    /* head > tail */
    {"gt:<fifo",        100,                     0,             256, true},
    {"gt:==fifo",       256,                     0,             256, false},
    {"gt:>fifo",        300,                     0,             256, false},
    {"EA gt:<fifo",     EA | 100,                0,             256, true},
    {"EA gt:==fifo",    EA | 256,                0,             256, false},
    {"EA gt:>fifo",     EA | 300,                0,             256, false},

    /* Large deltas around 2^31 to catch regressions to 32-bit compare */
    {"gt:d=2^31-1@t0",  POW2_31 - 1ull,          0,             256, false},
    {"gt:d=2^31@t0",    POW2_31,                 0,             256, false},
    {"gt:d=2^31+1@t0",  POW2_31 + 1ull,          0,             256, false},
    {"EA gt:d=2^31@t0", EA | POW2_31,            0,             256, false},

    /* head == tail */
    {"eq:zero",         0,                       0,             256, true},
    {"eq:EA",           EA,                      0,             256, true},

    /* Around 2^62 boundaries (head < tail deltas) */
    {"lt:2^62-1",       512,                     512 + LIM - 1, 256, true},
    {"lt:2^62",         512,                     512 + LIM,     256, true},
    {"lt:2^62+1",       512,                     512 + LIM + 1, 256, true},

    /* Special tail at MSB (robustness) */
    {"tailEA:+255",     0xff,                    EA,            256, true},
    {"tailEA:+256",     0x100,                   EA,            256, true},

    /* Practical cap at 2^62 */
    {"cap:eq",          LIM,                     LIM,           256, true},
    {"cap:eq EA",       EA | LIM,                LIM,           256, true}
};

}

class test_mm_fifo_room : public ucs::test {
protected:
    void check_case(const case_item &c) {
        bool got = UCT_MM_EP_IS_ABLE_TO_SEND(c.head, c.tail, c.fifo);
        EXPECT_EQ(c.expect, got) << c.name;
    }
};

UCS_TEST_F(test_mm_fifo_room, predicate_matrix) {
    for (const auto &c : k_cases) {
        check_case(c);
    }

}