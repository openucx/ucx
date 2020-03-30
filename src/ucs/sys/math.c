/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "math.h"

static uint64_t ucs_large_primes[] = {
    14476643271716824181ull, 12086978239110065677ull,
    15386586898367453843ull, 17958312454893560653ull,

    32416188191ull, 32416188793ull,
    32416189381ull, 32416190071ull,

    9929050057ull, 9929050081ull, 9929050097ull, 9929050111ull,
    9929050121ull, 9929050133ull, 9929050139ull, 9929050163ull,
    9929050207ull, 9929050217ull, 9929050249ull, 9929050253ull
};

uint64_t ucs_get_prime(unsigned index)
{
    static const unsigned num_primes = sizeof(ucs_large_primes) / sizeof(ucs_large_primes[0]);

    return ucs_large_primes[index % num_primes];
}
