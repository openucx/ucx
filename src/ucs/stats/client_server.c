/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "libstats.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#include <netdb.h>
#include <unistd.h>
#include <pthread.h>
#include <inttypes.h>
#include <sys/uio.h>

#include <ucs/datastruct/sglib_wrapper.h>
#include <ucs/sys/compiler.h>
#include <ucs/debug/log.h>

#define UCS_STATS_MAGIC            "UCSSTAT1"
#define UCS_STATS_MSG_FRAG_SIZE    1400
#define ENTITY_HASH_SIZE           997


/* UDP packet header */
typedef struct ucs_stats_packet_hdr {
    char                magic[8];
    uint64_t            timestamp;
    uint32_t            total_size;
    uint32_t            frag_offset;
    uint32_t            frag_size;
} UCS_S_PACKED ucs_stats_packet_hdr_t;


/* Fragment assembly hole free-list */
typedef struct frag_hole {
    ucs_list_link_t     list;
    size_t              size; /* Including this struct */
} frag_hole_t;


/* An entity which reports statistics */
typedef struct stats_entity stats_entity_t;
struct stats_entity {
    struct sockaddr_in  in_addr;        /* Entity address */
    uint64_t            timestamp;      /* Current timestamp */
    size_t              buffer_size;    /* Buffer size */
    void                *inprogress_buffer;    /* Fragment assembly buffer */
    ucs_list_link_t     holes;          /* List of holes in the buffer */
    stats_entity_t      *next;          /* Hash link */

    pthread_mutex_t     lock;
    volatile unsigned   refcount;
    void                *completed_buffer;  /* Completed buffer */
    struct timeval      update_time;
};


/* Client context */
typedef struct ucs_stats_client {
    int              sockfd;
} ucs_stats_client_t;


/* Server context */
typedef struct ucs_stats_server {
    int                     sockfd;
    int                     udp_port;
    pthread_t               server_thread;
    volatile unsigned long  rcvd_packets;
    volatile int            stop;
    ucs_list_link_t         curr_stats;
    pthread_mutex_t         entities_lock;
    stats_entity_t*         entities_hash[ENTITY_HASH_SIZE];
} ucs_stats_server_t;


SGLIB_DEFINE_LIST_PROTOTYPES(stats_entity_t, stats_entity_cmp, next)
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(stats_entity_t, ENTITY_HASH_SIZE, stats_entity_hash)


ucs_status_t ucs_stats_client_init(const char *server_addr, int port, ucs_stats_client_h *p_client)
{
    ucs_stats_client_h client;
    struct sockaddr_in saddr;
    struct hostent *he;
    ucs_status_t status;
    int ret;

    client = malloc(sizeof *client);
    if (client == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    he = gethostbyname(server_addr);
    if (he == NULL || he->h_addr_list == NULL) {
        ucs_error("failed to resolve address of '%s'", server_addr);
        status = UCS_ERR_INVALID_ADDR;
        goto err_free;
    }

    saddr.sin_family = he->h_addrtype;
    saddr.sin_port   = htons(port);
    assert(he->h_length == sizeof(saddr.sin_addr));
    memcpy(&saddr.sin_addr, he->h_addr_list[0], he->h_length);
    memset(saddr.sin_zero, 0, sizeof(saddr.sin_zero));

    client->sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (client->sockfd < 0) {
        ucs_error("socket() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_free;
    }

    ret = connect(client->sockfd, (struct sockaddr *)&saddr, sizeof(saddr));
    if (ret < 0) {
        ucs_error("connect(%d) failed: %m", client->sockfd);
        status = UCS_ERR_IO_ERROR;
        goto err_close;
    }

    *p_client = client;
    return UCS_OK;

err_close:
    close(client->sockfd);
err_free:
    free(client);
err:
    return status;
}

void ucs_stats_client_cleanup(ucs_stats_client_h client)
{
    close(client->sockfd);
    free(client);
}

static ucs_status_t
ucs_stats_sock_send_frags(int sockfd, uint64_t timestamp, void *buffer, size_t size)
{
    struct iovec iov[2];
    ucs_stats_packet_hdr_t hdr;
    size_t frag_size, offset, nsent;
    size_t max_frag = UCS_STATS_MSG_FRAG_SIZE - sizeof(hdr);

    offset = 0;

    memcpy(hdr.magic, UCS_STATS_MAGIC, sizeof(hdr.magic));
    hdr.total_size  = size;
    hdr.timestamp   = timestamp;

    while (offset < size) {
        frag_size = size - offset;
        if (frag_size > max_frag) {
            frag_size = max_frag;
        }

        hdr.frag_offset = offset;
        hdr.frag_size   = frag_size;

        iov[0].iov_base = &hdr;
        iov[0].iov_len  = sizeof(hdr);
        iov[1].iov_base = buffer + offset;
        iov[1].iov_len  = hdr.frag_size;

        nsent = writev(sockfd, iov, 2);
        if (nsent == -1) {
            if (errno == ECONNREFUSED) {
                ucs_trace("stats server is down");
                return UCS_OK;
            } else {
                ucs_error("writev() failed: %m");
                return UCS_ERR_IO_ERROR;
            }
        }

        assert(nsent == sizeof(hdr) + frag_size);
        offset += frag_size;
    }

    return UCS_OK;
}

ucs_status_t
ucs_stats_client_send(ucs_stats_client_h client, ucs_stats_node_t *root,
                      uint64_t timestamp)
{
    ucs_status_t status;
    FILE *stream;
    char *buffer;
    size_t size;

    /* TODO use GLIBC custom stream */
    stream = open_memstream(&buffer, &size);
    if (stream == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    status = ucs_stats_serialize(stream, root, UCS_STATS_SERIALIZE_BINARY);
    fclose(stream);

    if (status != UCS_OK) {
        goto out_free;
    }

    /* send */
    status = ucs_stats_sock_send_frags(client->sockfd, timestamp, buffer, size);

out_free:
    free(buffer);
out:
    return status;
}

static void ucs_stats_server_entity_reset_buffer(stats_entity_t * entity,
                                                 size_t new_size)
{
    frag_hole_t *hole;

    if (new_size != entity->buffer_size) {
        pthread_mutex_lock(&entity->lock);
        entity->buffer_size = new_size;
        entity->inprogress_buffer = realloc(entity->inprogress_buffer,
                                            new_size + sizeof(frag_hole_t));
        entity->completed_buffer  = realloc(entity->completed_buffer,
                                            new_size + sizeof(frag_hole_t));
        pthread_mutex_unlock(&entity->lock);
    }

    hole = entity->inprogress_buffer;
    hole->size = entity->buffer_size;
    ucs_list_head_init(&entity->holes);
    ucs_list_add_tail(&entity->holes, &hole->list);
}

static stats_entity_t *ucs_stats_server_entity_alloc(struct sockaddr_in *addr)
{
    stats_entity_t *entity;

    entity = malloc(sizeof *entity);
    if (entity == NULL) {
        return NULL;
    }

    entity->in_addr           = *addr;
    entity->timestamp         = 0;
    entity->buffer_size       = -1;
    entity->inprogress_buffer = NULL;
    entity->completed_buffer  = NULL;
    entity->refcount          = 1;
    ucs_list_head_init(&entity->holes);
    pthread_mutex_init(&entity->lock, NULL);

    ucs_stats_server_entity_reset_buffer(entity, 0);
    return entity;
}

static void ucs_stats_server_entity_free(stats_entity_t * entity)
{
    free(entity->inprogress_buffer);
    free(entity->completed_buffer);
    free(entity);
}

static stats_entity_t*
ucs_stats_server_entity_get(ucs_stats_server_h server, struct sockaddr_in *addr)
{
    stats_entity_t *entity, search;

    pthread_mutex_lock(&server->entities_lock);
    search.in_addr = *addr;

    entity = sglib_hashed_stats_entity_t_find_member(server->entities_hash, &search);
    if (entity == NULL) {
        entity = ucs_stats_server_entity_alloc(addr);
        gettimeofday(&entity->update_time, NULL);
        sglib_hashed_stats_entity_t_add(server->entities_hash, entity);
    }

    __sync_fetch_and_add(&entity->refcount, 1);
    pthread_mutex_unlock(&server->entities_lock);

    return entity;
}

static void ucs_stats_server_entity_put(stats_entity_t * entity)
{
    if (__sync_fetch_and_add(&entity->refcount, -1) == 1) {
        ucs_stats_server_entity_free(entity);
    }
}

/**
 * Find a hole to contain the given fragment.
 */
static frag_hole_t *
find_frag_hole(stats_entity_t *entity, size_t frag_size, size_t frag_offset)
{
    void *frag_start = entity->inprogress_buffer + frag_offset;
    void *frag_end   = entity->inprogress_buffer + frag_offset + frag_size;
    frag_hole_t *hole;

    ucs_list_for_each(hole, &entity->holes, list) {
        if ((frag_start >= (void*)hole) && (frag_end <= (void*)hole + hole->size)) {
            return hole;
        }
    }
    return NULL;
}

/**
 * Update statistics with new arrived fragment.
 */
static ucs_status_t
ucs_stats_server_entity_update(ucs_stats_server_h server, stats_entity_t *entity,
                               uint64_t timestamp, size_t total_size, void *frag,
                               size_t frag_size, size_t frag_offset)
{
    frag_hole_t *hole, *new_hole;
    void *frag_start, *frag_end, *hole_end;

    ucs_debug("From %s:%d - timestamp %"PRIu64", %zu..%zu / %zu",
              inet_ntoa(entity->in_addr.sin_addr), ntohs(entity->in_addr.sin_port),
              timestamp, frag_offset, frag_offset + frag_size, total_size);

    if (timestamp < entity->timestamp) {
        ucs_debug("Dropping - old timestamp");
        return 0;
    } else if (timestamp > entity->timestamp) {
        ucs_debug("New timestamp, resetting buffer with size %zu", total_size);
        entity->timestamp = timestamp;
        ucs_stats_server_entity_reset_buffer(entity, total_size);
    } else {
        /* Make sure all packets in this timestamp have the same 'total_size' */
        if (entity->buffer_size != total_size) {
            ucs_error("Total size in the packet is %zu, but expected is %zu",
                      total_size, entity->buffer_size);
        }
    }

    hole = find_frag_hole(entity, frag_size, frag_offset);
    if (hole == NULL) {
        ucs_error("cannot fill fragment (offset %zu size %zu)", frag_offset, frag_size);
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    frag_start = entity->inprogress_buffer + frag_offset;
    frag_end   = entity->inprogress_buffer + frag_offset + frag_size;
    hole_end   = (void*)hole + hole->size;

    ucs_debug("inserting into a hole of %zu..%zu",
              (void*)hole - entity->inprogress_buffer,
              hole_end    - entity->inprogress_buffer);

    /* If the fragment does not reach the end of the hole, create a new hole
     * in this space.
     */
    if (frag_end < hole_end) {
        /* Make sure we don't create a hole which is too small for a free-list
         * pointer to fit in. An exception is the last fragment.
         */
        assert((hole_end - frag_end >= sizeof(*new_hole)) ||
               (hole_end == entity->inprogress_buffer + entity->buffer_size));
        new_hole = frag_end;
        new_hole->size = hole_end - frag_end;
        ucs_list_insert_after(&hole->list, &new_hole->list);
    }

    /* If we have room before the fragment, resize the hole. Otherwise, delete it */
    if (frag_start > (void*)hole) {
        assert(frag_start - (void*)hole >= sizeof(*hole));
        hole->size = frag_start - (void*)hole;
    } else {
        ucs_list_del(&hole->list);
    }

    /* Copy the fragment */
    memcpy(frag_start, frag, frag_size);

    /* Completed? */
    if (ucs_list_is_empty(&entity->holes)) {
        ucs_debug("timestamp %"PRIu64" fully assembled", entity->timestamp);
        pthread_mutex_lock(&entity->lock);
        memcpy(entity->completed_buffer, entity->inprogress_buffer, entity->buffer_size);
        pthread_mutex_unlock(&entity->lock);
    }

    return UCS_OK;
}

/**
 * Update context with new arrived packet.
 */
static ucs_status_t
ucs_stats_server_update_context(ucs_stats_server_h server, struct sockaddr_in *sender,
                                ucs_stats_packet_hdr_t *pkt, size_t pkt_len)
{
    stats_entity_t *entity;
    ucs_status_t status;

    /* Validate fragment size */
    if (pkt_len != pkt->frag_size + sizeof(ucs_stats_packet_hdr_t)) {
        ucs_error("Invalid receive size: expected %zu, got %zu",
                  pkt->frag_size + sizeof(ucs_stats_packet_hdr_t), pkt_len);
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    /* Validate magic */
    if (memcmp(pkt->magic, UCS_STATS_MAGIC, sizeof(pkt->magic)) != 0) {
        ucs_error("Invalid magic in packet header");
        return UCS_ERR_INVALID_PARAM;
    }

    /* Find or create the entity */
    entity = ucs_stats_server_entity_get(server, sender);

    pthread_mutex_lock(&entity->lock);
    gettimeofday(&entity->update_time, NULL);
    pthread_mutex_unlock(&entity->lock);

    /* Update the entity */
    status = ucs_stats_server_entity_update(server, entity, pkt->timestamp,
                                           pkt->total_size, pkt + 1,
                                           pkt->frag_size, pkt->frag_offset);

    ucs_stats_server_entity_put(entity);
    ++server->rcvd_packets;
    return status;
}

static ucs_status_t ucs_stats_server_create_socket(int udp_port, int *p_sockfd,
                                                  int *p_udp_port)
{
    struct sockaddr_in saddr;
    socklen_t socklen;
    int sockfd;
    int ret;

    sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sockfd < 0) {
        ucs_error("socked() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    saddr.sin_family      = AF_INET;
    saddr.sin_addr.s_addr = INADDR_ANY;
    saddr.sin_port        = htons(udp_port);
    memset(saddr.sin_zero, 0, sizeof(saddr.sin_zero));

    ret = bind(sockfd, (struct sockaddr*)&saddr, sizeof(saddr));
    if (ret < 0) {
        ucs_error("Failed to bind socket to port %u: %m", udp_port);
        goto err_close_sock;
    }

    socklen = sizeof(saddr);
    ret = getsockname(sockfd, (struct sockaddr*)&saddr, &socklen);
    if (ret < 0) {
        ucs_error("getsockname(%d) failed: %m", sockfd);
        goto err_close_sock;
    }

    *p_sockfd   = sockfd;
    *p_udp_port = ntohs(saddr.sin_port);
    return UCS_OK;

err_close_sock:
    close(sockfd);
    return UCS_ERR_INVALID_ADDR;
}

static void ucs_stats_server_clear_old_enitities(ucs_stats_server_h server)
{
    struct sglib_hashed_stats_entity_t_iterator it;
    stats_entity_t *entity;
    struct timeval current, diff;

    gettimeofday(&current, NULL);

    pthread_mutex_lock(&server->entities_lock);
    entity = sglib_hashed_stats_entity_t_it_init(&it,server->entities_hash);
    while (entity != NULL) {
        pthread_mutex_lock(&entity->lock);
        timersub(&current, &entity->update_time, &diff);
        pthread_mutex_unlock(&entity->lock);

        if (diff.tv_sec > 5.0) {
            sglib_hashed_stats_entity_t_delete(server->entities_hash, entity);
            ucs_stats_server_entity_put(entity);
        }
        entity = sglib_hashed_stats_entity_t_it_next(&it);
     }

     pthread_mutex_unlock(&server->entities_lock);
}

static void* ucs_stats_server_thread_func(void *arg)
{
    ucs_stats_server_h server = arg;
    struct sockaddr_in recv_addr;
    socklen_t recv_addr_len;
    char recv_buf[UCS_STATS_MSG_FRAG_SIZE];
    ssize_t recv_len;
    ucs_status_t status;

    ucs_debug("starting server thread");
    while (!server->stop) {
        recv_addr_len = sizeof(recv_addr);
        recv_len = recvfrom(server->sockfd, recv_buf, UCS_STATS_MSG_FRAG_SIZE, 0,
                            (struct sockaddr*)&recv_addr, &recv_addr_len);
        if (recv_len < 0) {
            ucs_error("recvfrom() failed: %s (return value: %ld)", strerror(errno),
                      recv_len);
            break;
        } else if (recv_len == 0) {
            ucs_debug("Empty receive - ignoring");
            continue;
        }

        if (recv_addr.sin_family != AF_INET) {
            ucs_error("invalid address family from recvfrom()");
            break;
        }

        /* Update with new data */
        /* coverity[tainted_data] */
        status = ucs_stats_server_update_context(server, &recv_addr, (void*)recv_buf, recv_len);
        if (status != UCS_OK) {
            break;
        }

        ucs_stats_server_clear_old_enitities(server);
    }

    ucs_debug("terminating server thread");
    return NULL;
}

ucs_status_t ucs_stats_server_start(int port, ucs_stats_server_h *p_server)
{
    ucs_stats_server_h server;
    ucs_status_t status;

    server = malloc(sizeof *server);
    if (server == NULL) {
        ucs_error("Failed to allocate stats context");
        return UCS_ERR_NO_MEMORY;
    }

    pthread_mutex_init(&server->entities_lock, NULL);
    ucs_list_head_init(&server->curr_stats);
    sglib_hashed_stats_entity_t_init(server->entities_hash);

    status = ucs_stats_server_create_socket(port, &server->sockfd, &server->udp_port);
    if (status != UCS_OK) {
        free(server);
        return status;
    }

    server->rcvd_packets = 0;
    server->stop         = 0;
    pthread_create(&server->server_thread, NULL, ucs_stats_server_thread_func,
                   server);

    *p_server = server;
    return UCS_OK;
}

void ucs_stats_server_destroy(ucs_stats_server_h server)
{
    struct sglib_hashed_stats_entity_t_iterator it;
    stats_entity_t *entity;
    void *retval;

    server->stop = 1;
    shutdown(server->sockfd, SHUT_RDWR);
    pthread_join(server->server_thread, &retval);
    close(server->sockfd);

    ucs_stats_server_purge_stats(server);

    entity = sglib_hashed_stats_entity_t_it_init(&it,server->entities_hash);
    while (entity != NULL) {
        ucs_stats_server_entity_put(entity);
        entity = sglib_hashed_stats_entity_t_it_next(&it);
    }
    free(server);
}

int ucs_stats_server_get_port(ucs_stats_server_h server)
{
   return server->udp_port;
}

ucs_list_link_t *ucs_stats_server_get_stats(ucs_stats_server_h server)
{
    struct sglib_hashed_stats_entity_t_iterator it;
    stats_entity_t *entity;
    ucs_stats_node_t *node;
    ucs_status_t status;
    FILE *stream;

    ucs_stats_server_purge_stats(server);

    pthread_mutex_lock(&server->entities_lock);
    for (entity = sglib_hashed_stats_entity_t_it_init(&it, server->entities_hash);
         entity != NULL; entity = sglib_hashed_stats_entity_t_it_next(&it))
    {
        /* Parse the statistics data */
        pthread_mutex_lock(&entity->lock);
        stream = fmemopen(entity->completed_buffer, entity->buffer_size, "rb");
        status = ucs_stats_deserialize(stream, &node);
        fclose(stream);
        pthread_mutex_unlock(&entity->lock);

        if (status == UCS_OK) {
            ucs_list_add_tail(&server->curr_stats, &node->list);
        }
    }
    pthread_mutex_unlock(&server->entities_lock);

    return &server->curr_stats;
}

void ucs_stats_server_purge_stats(ucs_stats_server_h server)
{
    ucs_stats_node_t *node, *tmp;

    ucs_list_for_each_safe(node, tmp, &server->curr_stats, list) {
        ucs_list_del(&node->list);
        ucs_stats_free(node);
    }
}

unsigned long ucs_stats_server_rcvd_packets(ucs_stats_server_h server)
{
   return server->rcvd_packets;
}

static inline int stats_entity_cmp(stats_entity_t *e1, stats_entity_t *e2)
{
    int addr_diff = e1->in_addr.sin_addr.s_addr < e2->in_addr.sin_addr.s_addr;
    if (addr_diff != 0) {
        return addr_diff;
    } else {
        return ntohs(e1->in_addr.sin_port) - ntohs(e1->in_addr.sin_port);
    }
}

static inline int stats_entity_hash(stats_entity_t *e)
{
    return (((uint64_t)e->in_addr.sin_addr.s_addr << 16) + (uint64_t)ntohs(e->in_addr.sin_port)) % ENTITY_HASH_SIZE;
}

SGLIB_DEFINE_LIST_FUNCTIONS(stats_entity_t, stats_entity_cmp, next)
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(stats_entity_t, ENTITY_HASH_SIZE, stats_entity_hash)
