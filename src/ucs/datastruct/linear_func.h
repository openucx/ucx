/*
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_LINEAR_FUNC_H_
#define UCS_LINEAR_FUNC_H_

#include <ucs/sys/compiler_def.h>


/**
 * A 1d linear function, represented as f(x) = c + x * m.
 */
typedef struct {
    double         c;  /* constant factor */
    double         m;  /* multiplicative factor */
} ucs_linear_func_t;


/**
 * Calculate the linear function value for a specific point.
 *
 * @param [in] func    Linear function to apply.
 * @param [in] x       Point to apply the function at.
 *
 * @return f(x)
 */
static UCS_F_ALWAYS_INLINE double
ucs_linear_func_apply(const ucs_linear_func_t *func, double x)
{
    return func->c + (func->m * x);
}


/**
 * Sum two linear functions.
 *
 * @param [out] result   Filled with the resulting linear function.
 * @param [in]  func1    First function to add.
 * @param [in]  func2    Second function to add.
 */
static UCS_F_ALWAYS_INLINE void
ucs_linear_func_add(ucs_linear_func_t *result, const ucs_linear_func_t *func1,
                    const ucs_linear_func_t *func2)
{
    result->m = func1->m + func2->m;
    result->c = func1->c + func2->c;
}


#endif
