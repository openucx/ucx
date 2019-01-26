/**
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#include <stdint.h>
#include <ucp/api/ucp.h>
#include <sys/types.h>
#include <unistd.h>
#include "common.h"
#define HNAME_MAX_LEN 256
#define TAG_STR_MAX_LEN 512

typedef void (*listener_accept_cb_func)(void *client_ep_ptr, void *user_data);

struct ucx_context {
    int             completed;
};

typedef struct ucp_py_internal_ep {
    ucp_ep_h  *ep_ptr;
    char      ep_tag_str[TAG_STR_MAX_LEN];
    ucp_tag_t send_tag;
    ucp_tag_t recv_tag;
} ucp_py_internal_ep_t;

int ucp_py_init();
int ucp_py_listen(listener_accept_cb_func, void *, int);
int ucp_py_finalize(void);
void *ucp_py_get_ep(char *, int);
int ucp_py_put_ep(void *);

void ucp_py_worker_progress();
struct ucx_context *ucp_py_ep_send_nb(void *ep_ptr, struct data_buf *send_buf, int length);
struct ucx_context *ucp_py_recv_nb(void *ep_ptr, struct data_buf *buf, int length);
int ucp_py_ep_post_probe();
int ucp_py_probe_query(void *ep_ptr);
int ucp_py_probe_wait(void *ep_ptr);
int ucp_py_query_request(struct ucx_context *request);
