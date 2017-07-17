/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_WORKER_H_
#define UCT_WORKER_H_

#include <uct/api/uct.h>
#include <ucs/datastruct/callbackq.h>
#include <ucs/datastruct/list.h>


/**
 * Transport-specific data on a worker
 */
typedef struct uct_worker_tl_data {
    ucs_list_link_t        list;
    uint32_t               refcount;
    uint32_t               key;
    void                   *ptr;
} uct_worker_tl_data_t;


typedef struct uct_priv_worker {
    uct_worker_t           super;
    ucs_async_context_t    *async;
    ucs_thread_mode_t      thread_mode;
    ucs_list_link_t        tl_data;
} uct_priv_worker_t;


typedef struct uct_worker_progress {
    uct_worker_cb_id_t     id;
    uint32_t               refcount;
} uct_worker_progress_t;


#define uct_worker_tl_data_get(_worker, _key, _type, _cmp_fn, _init_fn, ...) \
    ({ \
        uct_worker_tl_data_t *data; \
        \
        ucs_list_for_each(data, &(_worker)->tl_data, list) { \
            if ((data->key == (_key)) && _cmp_fn(ucs_derived_of(data, _type), \
                                                 ## __VA_ARGS__)) \
            { \
                ++data->refcount; \
                break; \
            } \
        } \
        \
        if (&data->list == &(_worker)->tl_data) { \
            data = ucs_malloc(sizeof(_type), UCS_PP_QUOTE(_type)); \
            if (data != NULL) { \
                data->key      = (_key); \
                data->refcount = 1; \
                _init_fn(ucs_derived_of(data, _type), ## __VA_ARGS__); \
                ucs_list_add_tail(&(_worker)->tl_data, &data->list); \
            } \
        } \
        ucs_derived_of(data, _type); \
    })


#define uct_worker_tl_data_put(_data, _cleanup_fn, ...) \
    { \
        uct_worker_tl_data_t *data = (uct_worker_tl_data_t*)(_data); \
        if (--data->refcount == 0) { \
            ucs_list_del(&data->list); \
            _cleanup_fn((_data), ## __VA_ARGS__); \
            ucs_free(data); \
        } \
    }


void uct_worker_progress_init(uct_worker_progress_t *prog);

void uct_worker_progress_add_safe(uct_priv_worker_t *worker, ucs_callback_t cb,
                                  void *arg, uct_worker_progress_t *prog);

void uct_worker_progress_remove_all(uct_priv_worker_t *worker,
                                    uct_worker_progress_t *prog);

void uct_worker_progress_remove(uct_priv_worker_t *worker, uct_worker_progress_t *prog);

#endif
