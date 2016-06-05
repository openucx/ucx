/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
/*-
 * Copyright (c) 1992, 1993
 *      The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "qsort_r.h"

#include <ucs/sys/compiler.h>
#include <ucs/sys/math.h>
#include <stdlib.h>


/*
 * Qsort routine from Bentley & McIlroy's "Engineering a Sort Function".
 */
#define ucs_qsort_swapcode(TYPE, parmi, parmj, n)   \
    { \
        long i = (n) / sizeof (TYPE); \
        register TYPE *pi = (TYPE *) (parmi); \
        register TYPE *pj = (TYPE *) (parmj); \
        do { \
            register TYPE   t = *pi; \
            *pi++ = *pj; \
            *pj++ = t; \
        } while (--i > 0); \
    }

#define ucs_qsort_swaptype(a, size) \
    ({ \
        (((char *)a - (char *)0) % sizeof(long)) || \
            (size % sizeof(long)) ? 2 : (size == sizeof(long)) ? 0 : 1; \
    })

#define ucs_qsort_swap(a, b) \
    if (swaptype == 0) { \
        long t       = *(long *)(a); \
        *(long *)(a) = *(long *)(b); \
        *(long *)(b) = t; \
    } else { \
        ucs_qsort_swapfunc(a, b, size, swaptype); \
    }

#define ucs_qsort_vecswap(a, b, n) \
    if ((n) > 0) { \
        ucs_qsort_swapfunc(a, b, n, swaptype); \
    }

static UCS_F_ALWAYS_INLINE void
ucs_qsort_swapfunc(char *a, char *b, int n, int swaptype)
{
    if (swaptype <= 1) {
        ucs_qsort_swapcode(long, a, b, n)
    } else {
        ucs_qsort_swapcode(char, a, b, n)
    }
}

static UCS_F_ALWAYS_INLINE char *
ucs_qsort_med3(char *a, char *b, char *c, ucs_qsort_r_compare_cb_t *compare,
               void *arg)
{
    return (compare(a, b, arg) < 0) ?
              ((compare(b, c, arg) < 0) ? b : ((compare(a, c, arg)) < 0 ? c : a)) :
              ((compare(b, c, arg) > 0) ? b : ((compare(a, c, arg)) < 0 ? a : c));
}

void ucs_qsort_r(void *base, size_t nmemb, size_t size,
                 ucs_qsort_r_compare_cb_t *compare, void *arg)
{
    char *pa, *pb, *pc, *md, *pl, *pm, *pn;
    int d, r, swaptype, swap_cnt;

loop:
    swaptype = ucs_qsort_swaptype(base, size);
    swap_cnt = 0;

    if (nmemb < 7) {
        /* Switch to insertion sort */
        for (pm = (char*)base + size; pm < (char*)base + nmemb * size; pm += size) {
            for (pl = pm; pl > (char*)base && compare(pl - size, pl, arg) > 0; pl -= size) {
                ucs_qsort_swap(pl, pl - size);
            }
        }
        return;
    }

    pm = (char*)base + (nmemb / 2) * size;
    if (nmemb > 7) {
        pl = base;
        pn = (char*)base + (nmemb - 1) * size;
        if (nmemb > 40) {
            d = (nmemb / 8) * size;
            pl = ucs_qsort_med3(pl,         pl + d, pl + 2 * d, compare, arg);
            pm = ucs_qsort_med3(pm - d,     pm,     pm + d,     compare, arg);
            pn = ucs_qsort_med3(pn - 2 * d, pn - d, pn,         compare, arg);
        }
        pm = ucs_qsort_med3(pl, pm, pn, compare, arg);
    }

    ucs_qsort_swap(base, pm);
    pa = pb = (char*)base + size;

    pc = md = (char*)base + (nmemb - 1) * size;
    for (;;) {
        while ((pb <= pc) && (r = compare(pb, base, arg)) <= 0) {
            if (r == 0) {
                swap_cnt = 1;
                ucs_qsort_swap(pa, pb);
                pa += size;
            }
            pb += size;
        }
        while ((pb <= pc) && (r = compare(pc, base, arg)) >= 0) {
            if (r == 0) {
                swap_cnt = 1;
                ucs_qsort_swap(pc, md);
                md -= size;
            }
            pc -= size;
        }
        if (pb > pc) {
            break;
        }
        ucs_qsort_swap(pb, pc);
        swap_cnt = 1;
        pb += size;
        pc -= size;
    }

    if (swap_cnt == 0) {
        /* Switch to insertion sort */
        for (pm = (char*)base + size; pm < (char*)base + nmemb * size; pm += size) {
            for (pl = pm; pl > (char *)base && compare(pl - size, pl, arg) > 0;
                            pl -= size) {
                ucs_qsort_swap(pl, pl - size);
            }
        }
        return;
    }

    pn = (char*)base + nmemb * size;
    r  = ucs_min(pa - (char*)base, pb - pa);
    ucs_qsort_vecswap(base, pb - r, r);

    r  = ucs_min(md - pc, pn - md - size);
    ucs_qsort_vecswap(pb, pn - r, r);

    if ((r = pb - pa) > size) {
        ucs_qsort_r(base, r / size, size, compare, arg);
    }

    if ((r = md - pc) > size) {
        /* Iterate rather than recurse to save stack space */
        base  = pn - r;
        nmemb = r / size;
        goto loop;
    }
}
