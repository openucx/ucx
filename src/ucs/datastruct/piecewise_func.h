/*
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_PIECEWISE_FUNC_H_
#define UCS_PIECEWISE_FUNC_H_

#include <ucs/datastruct/linear_func.h>
#include <ucs/datastruct/list.h>
#include <stddef.h>


/**
 * Iterate over the segments of the piecewise function.
 *
 * @param [in]  _pw_func    Piecewise function to iterate over.
 * @param [out] _seg        Pointer variable to the current function segment.
 *
 */
#define ucs_piecewise_func_seg_foreach(_pw_func, _seg) \
    ucs_list_for_each((_seg), &(_pw_func)->head, list)


/*
 * A piecewise func segment which represents linear function on the range.
 * Start of the segment equals end of the previous segment + 1. The first
 * segment starts at 0.
 */
typedef struct {
    ucs_linear_func_t func; /* Function that applies on the segment range */
    size_t            end;  /* End of the segment (inclusive) */
    ucs_list_link_t   list; /* List entry */
} ucs_piecewise_segment_t;


/**
 * A piecewise function consisting of one or more segments.
 */
typedef struct {
    /* List of segments (list of the initialized function is always non-empty) */
    ucs_list_link_t head;
} ucs_piecewise_func_t;


/**
 * Initialize a piecewise function which represents f(x) = 0 on the
 * [0, SIZE_MAX] range.
 *
 * @param [in] pw_func    Piecewise function to initialize.
 *
 * @return UCS_OK in case of success, error otherwise.
 */
ucs_status_t ucs_piecewise_func_init(ucs_piecewise_func_t *pw_func);


/**
 * Free a piecewise function allocated memory.
 *
 * @param [in] pw_func    Piecewise function to free.
 */
void ucs_piecewise_func_cleanup(ucs_piecewise_func_t *pw_func);


/**
 * Calculate the piecewise function value for a specific point.
 *
 * @param [in] pw_func    Piecewise function to apply.
 * @param [in] x          Point to apply the function at.
 *
 * @return The value of the piecewise function at the given point.
 */
double ucs_piecewise_func_apply(const ucs_piecewise_func_t *pw_func, size_t x);


/**
 * Add a given linear function on the provided range of the piecewise function.
 *
 * @param [inout]  pw_func       Piecewise function to update.
 * @param [in]     start         Start of the range(inclusive).
 * @param [in]     end           End of the range(inclusive).
 * @param [in]     range_func    Linear function to add.
 *
 * @return UCS_OK in case of success, error otherwise.
 */
ucs_status_t ucs_piecewise_func_add_range(ucs_piecewise_func_t *pw_func,
                                          size_t start, size_t end,
                                          ucs_linear_func_t range_func);


/**
 * Add one piecewise function to another in-place.
 *
 * @param [inout]  dst_pw_func    First sum operand (result of the operation).
 * @param [in]     src_pw_func    Second sum operand.
 *
 * @return UCS_OK in case of success, error otherwise.
 */
ucs_status_t ucs_piecewise_func_add_inplace(ucs_piecewise_func_t *dst_pw_func,
                                            ucs_piecewise_func_t *src_pw_func);

#endif
