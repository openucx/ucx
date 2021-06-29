/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_CM_H_
#define UCT_CM_H_

#include <uct/api/uct_def.h>
#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>
#include <ucs/type/class.h>


UCS_CLASS_DECLARE(uct_listener_t, uct_cm_h);

#define UCT_CM_SET_CB(_params, _flag, _cep_cb, _params_cb, _function_type, _default_cb) \
    ({ \
        ucs_status_t _status; \
        \
        if (!((_params)->field_mask & (_flag))) { \
            (_cep_cb) = (_function_type) (_default_cb); \
            _status = UCS_OK; \
        } else if ((_params_cb) == NULL) { \
            ucs_error(UCS_PP_MAKE_STRING(_flag) " is set but the callback is NULL"); \
            _status = UCS_ERR_INVALID_PARAM; \
        } else { \
            (_cep_cb) = (_params_cb); \
            _status = UCS_OK; \
        } \
        \
        (_status); \
    })


#define uct_cm_peer_error(_cm, _fmt, ...) \
    { \
        ucs_log((_cm)->config.failure_level, _fmt, ## __VA_ARGS__); \
    }


#define uct_cm_ep_peer_error(_cep, _fmt, ...) \
    { \
        uct_cm_t *_cm_base = ucs_container_of((_cep)->super.super.iface, \
                                              uct_cm_t, iface); \
        uct_cm_peer_error(_cm_base, _fmt, ## __VA_ARGS__); \
    }


/**
 * "Base" structure which defines CM configuration options.
 * Specific CMs extend this structure.
 */
struct uct_cm_config {
    int          failure;   /* Level of failure reports */
    int          reuse_addr;
};

/**
 * Connection manager component operations
 */
typedef struct uct_cm_ops {
    void         (*close)(uct_cm_h cm);
    ucs_status_t (*cm_query)(uct_cm_h cm, uct_cm_attr_t *cm_attr);
    ucs_status_t (*listener_create)(uct_cm_h cm, const struct sockaddr *saddr,
                                    socklen_t socklen,
                                    const uct_listener_params_t *params,
                                    uct_listener_h *listener_p);
    ucs_status_t (*listener_reject)(uct_listener_h listener,
                                    uct_conn_request_h conn_request);
    ucs_status_t (*listener_query) (uct_listener_h listener,
                                    uct_listener_attr_t *listener_attr);
    void         (*listener_destroy)(uct_listener_h listener);
    ucs_status_t (*ep_create)(const uct_ep_params_t *params, uct_ep_h *ep_p);
} uct_cm_ops_t;


struct uct_cm {
    uct_cm_ops_t         *ops;
    uct_component_h      component;
    uct_base_iface_t     iface;

    struct {
        ucs_log_level_t  failure_level;
        int              reuse_addr;
    } config;
};


/**
 * Connection manager base endpoint
 */
typedef struct uct_cm_base_ep {
    uct_base_ep_t                       super;

    /* User data associated with the endpoint */
    void                                *user_data;

    /* Callback to handle the disconnection of the remote peer */
    uct_ep_disconnect_cb_t              disconnect_cb;

    /* Callback to fill the user's private data */
    uct_cm_ep_priv_data_pack_callback_t priv_pack_cb;

    /* Callback to notify bound device */
    uct_cm_ep_resolve_callback_t        resolve_cb;

    union {
        struct {
            /* On the client side - callback to process an incoming
             * connection response from the server */
            uct_cm_ep_client_connect_callback_t      connect_cb;
        } client;
        struct {
            /* On the server side - callback to process an incoming connection
             * establishment notification from the client */
            uct_cm_ep_server_conn_notify_callback_t  notify_cb;
        } server;
    };
} uct_cm_base_ep_t;


UCS_CLASS_DECLARE(uct_cm_base_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_NEW_FUNC(uct_cm_base_ep_t, uct_base_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_cm_base_ep_t, uct_base_ep_t);


extern ucs_config_field_t uct_cm_config_table[];

UCS_CLASS_DECLARE(uct_cm_t, uct_cm_ops_t*, uct_iface_ops_t*, uct_iface_internal_ops_t*,
                  uct_worker_h, uct_component_h, const uct_cm_config_t*);

ucs_status_t uct_listener_backlog_adjust(const uct_listener_params_t *params,
                                         int max_value, int *backlog);

ucs_status_t uct_cm_ep_set_common_data(uct_cm_base_ep_t *ep,
                                       const uct_ep_params_t *params);

ucs_status_t uct_cm_ep_pack_cb(uct_cm_base_ep_t *cep, void *arg,
                               const uct_cm_ep_priv_data_pack_args_t *pack_args,
                               void *priv_data, size_t priv_data_max,
                               size_t *priv_data_ret);

ucs_status_t uct_cm_ep_resolve_cb(uct_cm_base_ep_t *cep,
                                  const uct_cm_ep_resolve_args_t *args);

void uct_cm_ep_disconnect_cb(uct_cm_base_ep_t *cep);

void uct_cm_ep_client_connect_cb(uct_cm_base_ep_t *cep,
                                 uct_cm_remote_data_t *remote_data,
                                 ucs_status_t status);

void uct_cm_ep_server_conn_notify_cb(uct_cm_base_ep_t *cep, ucs_status_t status);

void uct_ep_connect_params_get(const uct_ep_connect_params_t *params,
                               const void **priv_data_p,
                               size_t *priv_data_length_p);

#endif /* UCT_CM_H_ */
