/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
*
* See file LICENSE for terms.
*/

#ifndef UCS_QUEUE_TYPES_H_
#define UCS_QUEUE_TYPES_H_


typedef struct ucs_queue_elem ucs_queue_elem_t;
typedef struct ucs_queue_head ucs_queue_head_t;
typedef ucs_queue_elem_t**    ucs_queue_iter_t;


/**
 * Queue element type.
 */
struct ucs_queue_elem {
    ucs_queue_elem_t    *next;
};


/**
 * Queue type.
 */
struct ucs_queue_head {
    ucs_queue_elem_t    *head;
    ucs_queue_elem_t    **ptail;
};


#endif
