/*
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_LINEAR_FUNC_H_
#define UCS_LINEAR_FUNC_H_

#include <ucs/type/status.h>
#include <ucs/sys/compiler_def.h>
#include <math.h>


/**
 * A 1d linear function, represented as f(x) = c + x * m.
 */
typedef struct {
    double         c;  /* constant factor */
    double         m;  /* multiplicative factor */
} ucs_linear_func_t;


/**
 * Construct a linear function
 *
 * @param [in]  c  Linear function constant functor
 * @param [in]  m  Linear function multiplicative functor
 *
 * @return A linear function which represents f(x) = c + x * m
 */
static UCS_F_ALWAYS_INLINE ucs_linear_func_t
ucs_linear_func_make(double c, double m)
{
    ucs_linear_func_t result;

    result.c = c;
    result.m = m;

    return result;
}


/**
 * Calculate the linear function value for a specific point.
 *
 * @param [in] func    Linear function to apply.
 * @param [in] x       Point to apply the function at.
 *
 * @return f(x)
 */
static UCS_F_ALWAYS_INLINE double
ucs_linear_func_apply(ucs_linear_func_t f, double x)
{
    return f.c + (f.m * x);
}


/**
 * Sum two linear functions.
 *
 * @param [in]  func1    First function to add.
 * @param [in]  func2    Second function to add.
 *
 * @return Linear function representing (func1 + func2)
 */
static UCS_F_ALWAYS_INLINE ucs_linear_func_t
ucs_linear_func_add(ucs_linear_func_t func1, ucs_linear_func_t func2)
{
    return ucs_linear_func_make(func1.c + func2.c, func1.m + func2.m);
}


/**
 * Sum three linear functions.
 *
 * @param [in]  func1    First function to add.
 * @param [in]  func2    Second function to add.
 * @param [in]  func3    Third function to add.
 *
 * @return Linear function representing (func1 + func2 + func3)
 */
static UCS_F_ALWAYS_INLINE ucs_linear_func_t
ucs_linear_func_add3(ucs_linear_func_t func1, ucs_linear_func_t func2,
                     ucs_linear_func_t func3)
{
    return ucs_linear_func_add(ucs_linear_func_add(func1, func2), func3);
}


/**
 * Subtract one linear function from another.
 *
 * @param [in]  func1    Linear function to subtract from.
 * @param [in]  func2    Linear function to subtract.
 *
 * @return Linear function representing (func1 - func2)
 */
static inline ucs_linear_func_t
ucs_linear_func_sub(ucs_linear_func_t func1, ucs_linear_func_t func2)
{
    return ucs_linear_func_make(func1.c - func2.c, func1.m - func2.m);
}


/**
 * Add one linear function to another in-place.
 *
 * @param [inout]  func1    First linear function to add, and the result of the
 *                          operation
 * @param [in]     func2    Second function to add.
 */
static inline void
ucs_linear_func_add_inplace(ucs_linear_func_t *func1, ucs_linear_func_t func2)
{
    func1->m += func2.m;
    func1->c += func2.c;
}


/**
 * Substitute the "x" argument of a linear function by another linear function
 * and return the composition of the functions.
 *
 * @param [in]  outer  Linear function whose "x" argument should be substituted.
 * @param [in]  inner  Linear function to substitute.
 *
 * @return Linear function representing outer(inner(x))
 */
static inline ucs_linear_func_t
ucs_linear_func_compose(ucs_linear_func_t outer, ucs_linear_func_t inner)
{
    /*
     * let  outer(x) = outer.m*x + outer.c, and g(x) = g.m*x + g.c
     * then outer(g(x)) = outer.m(inner.m*x + g.c) + outer.c =
     *                    (outer.m * inner.m)x + (outer.m*g.c + outer.c)
     */
    return ucs_linear_func_make((outer.m * inner.c) + outer.c,
                                outer.m * inner.m);
}


/**
 * Find the intersection point between two linear functions. If the functions
 * do not intersect, the result is undefined.
 *
 * @param [in]   func1        First function to intersect.
 * @param [in]   func2        Second function to intersect.
 * @param [out]  x_intersect  Upon success, set to the X-value of the
 *                            intersection point.
 *
 * @return UCS_OK if success, UCS_ERR_INVALID_PARAM if the linear functions have
 *         no intersection, or if their intersection point exceeds the maximal
 *         double value.
 *
 */
static inline ucs_status_t
ucs_linear_func_intersect(ucs_linear_func_t func1, ucs_linear_func_t func2,
                          double *x_intersect)
{
    double x;

    x = (func2.c - func1.c) / (func1.m - func2.m);
    if (isnan(x) || isinf(x)) {
        return UCS_ERR_INVALID_PARAM;
    }

    *x_intersect = x;
    return UCS_OK;
}


/*
 * Increase the constant of a given linear function by a value of another linear
 * function at a specific point.
 *
 * @param [inout] func           Increase the constant of this linear function.
 * @param [in]    baseline_func  Add the value of this linear function.
 * @param [in]    baseline_x     Point at which to take the value of
 *                               @a baseline_func.
 */
static inline void
ucs_linear_func_add_value_at(ucs_linear_func_t *func,
                             ucs_linear_func_t baseline_func, double baseline_x)
{
    func->c += ucs_linear_func_apply(baseline_func, baseline_x);
}


/*
 * Check if two linear functions are equal.
 *
 * @param [in] func1    First function to compare.
 * @param [in] func2    Second function to compare.
 * @param [in] epsilon  Threshold to consider two floating-point values as equal.
 */
static inline int
ucs_linear_func_is_equal(ucs_linear_func_t func1, ucs_linear_func_t func2,
                         double epsilon)
{
    return (fabs(func1.m - func2.m) < epsilon) &&
           (fabs(func1.c - func2.c) < epsilon);
}

#endif
