/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023-2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/sock.h>
#include <ucs/debug/log.h>
#include <ucs/sys/iovec.inl>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/poll.h>
#include <linux/types.h> /* __be64 */

#include <infiniband/mad.h>
#include <infiniband/umad.h>
#include <infiniband/umad_types.h>

#include "../perftest_context.h"

#define PERFTEST_RTE_CLASS      (IB_VENDOR_RANGE2_START_CLASS + 0x10)
#define PERFTEST_RTE_MAD_QP_NUM 1 /* Don't use MAD on QP0 to not disturb SMI */

typedef struct perftest_mad_rte_group {
    struct ibmad_port *mad_port;
    ib_portid_t       dst_port;
    int               is_server;
} perftest_mad_rte_group_t;

static unsigned mad_magic = 0xdeadbeef;

static ucs_status_t
perftest_mad_get_remote_port(void *umad, ib_portid_t *remote_port)
{
    ib_mad_addr_t *mad_addr = umad_get_mad_addr(umad);

    if (mad_addr == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    if (ib_portid_set(remote_port, ntohs(mad_addr->lid), 0, 0)) {
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}

static ucs_status_t perftest_mad_sendv(perftest_mad_rte_group_t *mad,
                                       const struct iovec *iovec, int iovcnt)
{
    ib_rmpp_hdr_t rmpp = {
        /* Always active, even when data_size <= IB_VENDOR_RANGE2_DATA_SIZE */
        .flags = IB_RMPP_FLAG_ACTIVE,
    };
    ib_rpc_t rpc       = {};
    size_t data_size   = ucs_iovec_total_length(iovec, iovcnt);
    size_t size        = umad_size() + IB_VENDOR_RANGE2_DATA_OFFS + data_size;
    int umad_len, fd, agent, ret;
    ib_portid_t *portid;
    uint8_t *data;
    void *umad;
    ucs_status_t status;

    if (data_size > (INT_MAX - IB_VENDOR_RANGE2_DATA_OFFS)) {
        return UCS_ERR_INVALID_PARAM;
    }

    umad = calloc(1, size);
    if (umad == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    data = umad_get_mad(umad);
    data = UCS_PTR_BYTE_OFFSET(data, IB_VENDOR_RANGE2_DATA_OFFS);
    ucs_iov_copy(iovec, iovcnt, 0, data, data_size, UCS_IOV_COPY_TO_BUF);

    rpc.mgtclass = PERFTEST_RTE_CLASS;
    rpc.method   = IB_MAD_METHOD_TRAP;
    rpc.attr.id  = 0;
    rpc.attr.mod = 0;
    rpc.oui      = IB_OPENIB_OUI;
    rpc.timeout  = 0;
    rpc.dataoffs = 0;
    rpc.datasz   = 0; /* ok: mad_build_pkt() is passed NULL pointer */

    portid     = &mad->dst_port;
    portid->qp = PERFTEST_RTE_MAD_QP_NUM;
    if (portid->qkey == 0) {
        portid->qkey = IB_DEFAULT_QP1_QKEY;
    }

    ret = mad_build_pkt(umad, &rpc, &mad->dst_port, &rmpp, NULL);
    if (ret < 0) {
        ucs_error("mad_build_pkt(mgtclass=%d oui=%x) failed: %m", rpc.mgtclass,
                  rpc.oui);
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    agent    = mad_rpc_class_agent(mad->mad_port, rpc.mgtclass);
    fd       = mad_rpc_portid(mad->mad_port);
    umad_len = IB_VENDOR_RANGE2_DATA_OFFS + data_size;

    ret = umad_send(fd, agent, umad, umad_len, 0, 0);
    if (ret < 0) {
        ucs_error("umad_send(agent=%d umad_len=%d) failed: %m", agent,
                  umad_len);
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    status = UCS_OK;
out:
    free(umad);
    return status;
}

static ucs_status_t perftest_mad_send(perftest_mad_rte_group_t *rte_group,
                                      void *buffer, size_t size)
{
    const struct iovec iovec = {
        .iov_base = buffer,
        .iov_len  = size,
    };
    return perftest_mad_sendv(rte_group, &iovec, 1);
}

static const char ping_str[] = "ping";

static ucs_status_t perftest_mad_ping(perftest_mad_rte_group_t *rte_group)
{
    const struct iovec iovec = {
        .iov_base = (void*)ping_str,
        .iov_len  = sizeof(ping_str),
    };
    return perftest_mad_sendv(rte_group, &iovec, 1);
}

static int perftest_mad_is_ping(void *buffer, size_t size)
{
    size_t ping_size = sizeof(ping_str);
    if (size != ping_size) {
        return 0;
    }
    return memcmp(buffer, ping_str, ping_size) == 0;
}

static int perftest_mad_user_mad_read(int fd, struct ib_user_mad **umad_p)
{
    static const int timeout_msec = 3 * 1000;
    int umad_len                  = 32 * 1024; /* cannot use 'size_t' here */
    void *umad;
    ucs_status_t status;
    int ret;

    umad = malloc(umad_size() + umad_len);
    if (umad == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    ret = umad_recv(fd, umad, &umad_len, timeout_msec);
    if (ret >= 0) {
        ucs_assertv(umad_len >= 0, "umad_len: %d", umad_len);
        *umad_p = umad;
        return umad_len;
    }

    if ((errno == ETIMEDOUT) || (errno == EAGAIN)) {
        status = UCS_ERR_NO_PROGRESS;
    } else {
        status = UCS_ERR_IO_ERROR;
    }

    ucs_assertv(status < 0, "status: %d", status);
    free(umad);
    return status;
}

static ucs_status_t perftest_mad_recv(perftest_mad_rte_group_t *rte_group,
                                      void *buffer, size_t *avail,
                                      ib_portid_t *remote_port)
{
    int fd = mad_rpc_portid(rte_group->mad_port);
    int ret;
    uint8_t *data;
    int len;
    struct ib_user_mad *user_mad;
    ucs_status_t status;

ping_retry:
    len = perftest_mad_user_mad_read(fd, &user_mad);
    if (len < 0) {
        return len;
    }

    len -= IB_VENDOR_RANGE2_DATA_OFFS;
    if (len < 0) {
        if (user_mad->status == ETIMEDOUT) {
            ucs_error("MAD: receive: remote unreachable");
            status = UCS_ERR_UNREACHABLE;
        } else {
            status = UCS_ERR_OUT_OF_RANGE;
        }

        goto out;
    }

    status = perftest_mad_get_remote_port(user_mad, remote_port);
    if (status != UCS_OK) {
        goto out;
    }

    ret = umad_status(user_mad);
    if (ret != 0) {
        ucs_warn("MAD: receive: status failure: %d", ret);
        status = UCS_ERR_REJECTED;
        goto out;
    }

    data = UCS_PTR_BYTE_OFFSET(umad_get_mad(user_mad),
                               IB_VENDOR_RANGE2_DATA_OFFS);
    if (perftest_mad_is_ping(data, len)) {
        free(user_mad);
        goto ping_retry;
    }

    if (len > *avail) {
        status = UCS_ERR_MESSAGE_TRUNCATED;
        len    = *avail;
    } else {
        status = UCS_OK;
    }

    memcpy(buffer, data, len);
    *avail = len;

out:
    free(user_mad);
    return status;
}

static ucs_status_t
perftest_mad_recv_from_remote(perftest_mad_rte_group_t *rte_group, void *buffer,
                              size_t *avail, const ib_portid_t *target_port)
{
    ucs_status_t status = UCS_ERR_IO_ERROR;
    ib_portid_t remote_port;
    size_t size;

    for (;;) {
        size   = *avail;
        status = perftest_mad_recv(rte_group, buffer, &size, &remote_port);
        if (status == UCS_ERR_NO_PROGRESS) {
            status = perftest_mad_ping(rte_group);
            ucs_assertv(status == UCS_OK, "status: %s",
                        ucs_status_string(status));
            continue;
        }

        if (status != UCS_OK) {
            return status;
        }

        if (remote_port.lid == target_port->lid) {
            ucs_debug("MAD: recv packet size:%zu/%zu", size, *avail);
            *avail = size;
            return UCS_OK;
        }
    }
}

static unsigned rte_mad_group_size(void *rte_group)
{
    return 2;
}

static unsigned rte_mad_group_index(void *rte_group)
{
    return !((perftest_mad_rte_group_t*)rte_group)->is_server;
}

static ucs_status_t
perftest_mad_recv_magic(perftest_mad_rte_group_t *group, unsigned value)
{
    unsigned magic = 0;
    size_t size    = sizeof(magic);
    ucs_status_t status;

    status = perftest_mad_recv_from_remote(group, &magic, &size,
                                           &group->dst_port);
    if (status == UCS_ERR_UNREACHABLE) {
        return UCS_ERR_UNREACHABLE;
    }

    if ((status == UCS_OK) && (size == sizeof(magic)) && (magic == value)) {
        return UCS_OK;
    }

    ucs_debug("recv magic: magic 0x%08x, expected 0x%08x", magic, value);

    return UCS_ERR_REJECTED;
}

static ucs_status_t perftest_mad_barrier(perftest_mad_rte_group_t *group)
{
    ucs_status_t status = UCS_OK;
    unsigned value;

#if _OPENMP
#  pragma omp barrier
#  pragma omp single copyprivate(status)
#endif
    {
        value = ++mad_magic;

        /* Disambiguation: Server and Client send binary inverted values */
        if (group->is_server) {
            value = ~value;
        }

        status = perftest_mad_send(group, &value, sizeof(value));
        ucs_assertv(status == UCS_OK, "status: %s", ucs_status_string(status));

        status = perftest_mad_recv_magic(group, ~value);
    }
#if _OPENMP
#  pragma omp barrier
#endif
    return status;
}

static void
rte_mad_barrier(void *rte_group, void (*progress)(void *arg), void *arg)
{
    ucs_status_t status = perftest_mad_barrier(rte_group);
    if (status != UCS_OK) {
        ucs_error("MAD: rte barrier failure: %s", ucs_status_string(status));
        exit(EXIT_FAILURE);
    }
}

static void rte_mad_post_vec(void *rte_group, const struct iovec *iovec,
                             int iovcnt, void **req)
{
    ucs_status_t status = perftest_mad_sendv(rte_group, iovec, iovcnt);
    if (status != UCS_OK) {
        ucs_error("MAD: rte post failure: %s", ucs_status_string(status));
        exit(EXIT_FAILURE);
    }
}

static void
rte_mad_recv(void *rte_group, unsigned src, void *buffer, size_t max, void *req)
{
    perftest_mad_rte_group_t *group = rte_group;
    size_t size                     = max;
    ucs_status_t UCS_V_UNUSED status;

    if (src != group->is_server) {
        return;
    }

    status = perftest_mad_recv_from_remote(group, buffer, &size,
                                           &group->dst_port);
    if (status != UCS_OK) {
        ucs_error("MAD: rte recv failure: %s", ucs_status_string(status));
        exit(EXIT_FAILURE);
    }
}

static ucs_status_t rte_mad_setup(void *arg);
static void rte_mad_cleanup(void *arg);

static ucx_perf_rte_t mad_rte = {
    .setup        = rte_mad_setup,
    .cleanup      = rte_mad_cleanup,
    .group_size   = rte_mad_group_size,
    .group_index  = rte_mad_group_index,
    .barrier      = rte_mad_barrier,
    .post_vec     = rte_mad_post_vec,
    .recv         = rte_mad_recv,
    .exchange_vec = (ucx_perf_rte_exchange_vec_func_t)ucs_empty_function
};

static struct ibmad_port *perftest_mad_open(char *ca, int ca_port)
{
    int mgmt_classes[]     = {IB_SA_CLASS}; /* needed to activate RMPP */
    int mgmt_classes_size  = 1;
    int perftest_rte_class = PERFTEST_RTE_CLASS;
    int oui                = IB_OPENIB_OUI;
    int rmpp_version       = UMAD_RMPP_VERSION;
    struct ibmad_port *port;

    if ((ca == NULL) || (ca_port < 0)) {
        ucs_error("MAD: missing CA or CA port");
        return NULL;
    }

    port = mad_rpc_open_port(ca, ca_port, mgmt_classes, mgmt_classes_size);
    if (port == 0) {
        ucs_error("mad_rpc_open_port(ca=\"%s\" ca_port=%d "
                  "mgmt_classes=IB_SA_CLASS) failed: %m",
                  ca, ca_port);
        return NULL;
    }

    if (mad_register_server_via(perftest_rte_class, rmpp_version, NULL, oui,
                                port) < 0) {
        ucs_error("mad_register_server_via(mgmt=%d rmpp_version=%d oui=%d)"
                  " failed: %m",
                  IB_SA_CLASS, UMAD_RMPP_VERSION, oui);
        goto err;
    }

    return port;

err:
    mad_rpc_close_port(port);
    return NULL;
}

static ucs_status_t perftest_mad_path_query(const char *ca, int ca_port,
                                            const struct ibmad_port *mad_port,
                                            uint64_t guid,
                                            ib_portid_t *dst_port)
{
    uint8_t buf[IB_SA_DATA_SIZE] = {};
    char err_str[256];
    umad_port_t port;
    __be64 prefix;
    ibmad_gid_t selfgid;
    uint64_t port_guid;
    uint64_t gid_prefix;
    int ret;
    ib_portid_t sm_id; /* SM: the GUID to LID resolver */

    ret = umad_get_port(ca, ca_port, &port);
    if (ret < 0) {
        ucs_error("umad_get_port(ca=\"%s\" ca_port=%d) failed: %s", ca, ca_port,
                  strerror_r(-ret, err_str, sizeof(err_str)));
        return UCS_ERR_INVALID_PARAM;
    }

    memset(&sm_id, 0, sizeof(sm_id));
    sm_id.lid = port.sm_lid;
    sm_id.sl  = port.sm_sl;

    memset(selfgid, 0, sizeof(selfgid)); /* uint8_t[] */
    gid_prefix = be64toh(port.gid_prefix);
    port_guid  = be64toh(port.port_guid);

    umad_release_port(&port);

    mad_encode_field(selfgid, IB_GID_PREFIX_F, &gid_prefix);
    mad_encode_field(selfgid, IB_GID_GUID_F, &port_guid);

    memcpy(&prefix, selfgid, sizeof(prefix));
    mad_set_field64(dst_port->gid, 0, IB_GID_PREFIX_F,
                    prefix ? be64toh(prefix) : IB_DEFAULT_SUBN_PREFIX);
    mad_set_field64(dst_port->gid, 0, IB_GID_GUID_F, guid);

    dst_port->lid = ib_path_query_via(mad_port, selfgid, dst_port->gid, &sm_id,
                                      buf);
    if (dst_port->lid < 0) {
        ucs_error("MAD: GUID query failed");
        return UCS_ERR_UNREACHABLE;
    }

    mad_decode_field(buf, IB_SA_PR_SL_F, &dst_port->sl);
    return UCS_OK;
}

static ucs_status_t perftest_mad_get_portid(const char *ca, int ca_port,
                                            const char *addr,
                                            const struct ibmad_port *mad_port,
                                            ib_portid_t *dst_port)
{
    static const char guid_str[] = "guid:";
    static const char lid_str[]  = "lid:";
    int lid;
    uint64_t guid;

    memset(dst_port, 0, sizeof(*dst_port));

    /* Setup address and address type */
    if (!strncmp(addr, guid_str, strlen(guid_str))) {
        addr += strlen(guid_str);

        guid = strtoull(addr, NULL, 0);
        if (!guid) {
            return UCS_ERR_INVALID_PARAM;
        }

        return perftest_mad_path_query(ca, ca_port, mad_port, guid, dst_port);
    } else if (!strncmp(addr, lid_str, strlen(lid_str))) {
        addr += strlen(lid_str);

        lid = strtol(addr, NULL, 0);
        if (!IB_LID_VALID(lid)) {
            return UCS_ERR_INVALID_PARAM;
        }

        return ib_portid_set(dst_port, lid, 0, 0) ? UCS_ERR_NO_DEVICE : UCS_OK;
    }

    ucs_error("MAD: invalid dst address, use '%s' or '%s' prefix", guid_str,
              lid_str);
    return UCS_ERR_INVALID_PARAM;
}

static int perftest_mad_accept_is_valid(const void *buf, size_t size)
{
    const perftest_params_t *params = buf;
    size_t array_size;

    if (size < sizeof(*params)) {
        return 0;
    }

    array_size  = sizeof(*params->super.msg_size_list);
    array_size *= params->super.msg_size_cnt;

    return size == (sizeof(*params) + array_size);
}

static ucs_status_t perftest_mad_accept(perftest_mad_rte_group_t *rte_group,
                                        struct perftest_context *ctx)
{
    union {
        perftest_params_t params;
        uint8_t           buf[4096];
    } peer;
    size_t size;
    ucs_status_t status;
    int lid;

    do {
        size   = sizeof(peer.buf);
        status = perftest_mad_recv(rte_group, peer.buf, &size,
                                   &rte_group->dst_port);

        ucs_debug("MAD: accept: receive got status:%d, size:%zu/%zu", status,
                  size, sizeof(peer.buf));
    } while ((status != UCS_OK) || !perftest_mad_accept_is_valid(peer.buf, size));

    lid = rte_group->dst_port.lid;
    ucs_debug("MAD: accept: remote lid:%d/0x%02x", lid, lid);

    peer.params.super.msg_size_list =
            UCS_PTR_TYPE_OFFSET(&peer.params, peer.params);
    status = perftest_params_merge(&ctx->params, &peer.params);
    if (status != UCS_OK) {
        return status;
    }

    return perftest_mad_send(rte_group, &mad_magic, sizeof(mad_magic));
}

static ucs_status_t perftest_mad_connect(perftest_mad_rte_group_t *rte_group,
                                         struct perftest_context *ctx)
{
    ucs_status_t status;
    struct iovec iov[2];

    iov[0].iov_base = &ctx->params;
    iov[0].iov_len  = sizeof(ctx->params);
    iov[1].iov_base = ctx->params.super.msg_size_list;
    iov[1].iov_len  = sizeof(*ctx->params.super.msg_size_list) *
                      ctx->params.super.msg_size_cnt;

    status = perftest_mad_sendv(rte_group, iov, 2);
    if (status != UCS_OK) {
        return status;
    }

    return perftest_mad_recv_magic(rte_group, mad_magic);
}

static void perftest_mad_set_logging(void)
{
    static const int ib_debug_level = 10;

    if (ucs_log_is_enabled(UCS_LOG_LEVEL_DEBUG)) {
        ibdebug = ib_debug_level; /* extern variable from mad headers */
        umad_debug(ib_debug_level);
    }
}

static ucs_status_t perftest_mad_parse_ca_and_port(const char *mad_port,
                                                   char *ca, size_t ca_size,
                                                   int *ca_port)
{
    static const int default_port = 1;
    char *sep                     = strchr(mad_port, ':');
    size_t len;

    if (sep == NULL) {
        len      = strlen(mad_port);
        *ca_port = default_port;
    } else {
        len      = sep - mad_port;
        *ca_port = atoi(sep + 1);
    }

    if (len >= ca_size) {
        return UCS_ERR_INVALID_PARAM;
    }

    memcpy(ca, mad_port, len);
    ca[len] = '\0';
    return UCS_OK;
}

static ucs_status_t rte_mad_setup(void *arg)
{
    struct perftest_context *ctx = arg;
    perftest_mad_rte_group_t *rte_group;
    ucs_status_t status;
    int ca_port;
    char ca[32];

    if (ctx->mad_port == NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = perftest_mad_parse_ca_and_port(ctx->mad_port, ca, sizeof(ca),
                                            &ca_port);
    if (status != UCS_OK) {
        return status;
    }

    rte_group = calloc(1, sizeof(*rte_group));
    if (rte_group == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    perftest_mad_set_logging();

    rte_group->is_server = !ctx->server_addr;
    rte_group->mad_port  = perftest_mad_open(ca, ca_port);
    if (rte_group->mad_port == 0) {
        ucs_error("MAD: cannot open port: '%s:%d' -> '%s'", ca, ca_port,
                  ctx->server_addr);
        goto err;
    }

    if (rte_group->is_server) {
        status = perftest_mad_accept(rte_group, ctx);
        if (status != UCS_OK) {
            goto err_close_mad_port;
        }
    } else {
        /* Lookup server if needed */
        status = perftest_mad_get_portid(ca, ca_port, ctx->server_addr,
                                         rte_group->mad_port,
                                         &rte_group->dst_port);
        if (status != UCS_OK) {
            ucs_error("MAD: client: cannot get port as: '%s:%d' -> '%s'", ca,
                      ca_port, ctx->server_addr);
            goto err_close_mad_port;
        }

        /* Try to connect to it */
        status = perftest_mad_connect(rte_group, ctx);
        if (status != UCS_OK) {
            goto err_close_mad_port;
        }
    }

    ctx->params.super.rte_group  = rte_group;
    ctx->params.super.rte        = &mad_rte;

    if (rte_group->is_server) {
        ctx->flags |= TEST_FLAG_PRINT_TEST;
    } else {
        ctx->flags |= TEST_FLAG_PRINT_RESULTS;
    }

    return UCS_OK;

err_close_mad_port:
    mad_rpc_close_port(rte_group->mad_port);
err:
    free(rte_group);
    return UCS_ERR_NO_DEVICE;
}

static void rte_mad_cleanup(void *arg)
{
    struct perftest_context *ctx    = arg;
    perftest_mad_rte_group_t *group = ctx->params.super.rte_group;

    ctx->params.super.rte_group = NULL;
    if (group != NULL) {
        mad_rpc_close_port(group->mad_port);
        free(group);
    }
}

UCS_STATIC_INIT {
    ucs_list_add_head(&rte_list, &mad_rte.list);
}

UCS_STATIC_CLEANUP {
    ucs_list_del(&mad_rte.list);
}
