/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_CIRCULAR_BUF_H_
#define UCS_CIRCULAR_BUF_H_

typedef struct ucs_circular_buf {
    void *buf;
    int   head;
    int   tail;
    int   elem_size;
    int   num_elems;
} ucs_circular_buf_t;


static inline ucs_status_t
ucs_circular_buf_init(ucs_circular_buf_t *buf, int elem_size, int num_elems)
{
    buf->buf = ucs_calloc(num_elems, elem_size, "circ_buf");

    if (buf->buf == NULL) {
        ucs_error("Failed to allocate memory for circular buffer");
        return UCS_ERR_NO_MEMORY;
    }

    buf->head = buf->tail = 0;
    buf->num_elems = num_elems;
    buf->elem_size = elem_size;

    return UCS_OK;
}

static inline void
ucs_circular_buf_cleanup(ucs_circular_buf_t *buf)
{
    ucs_free(buf->buf);
}

static inline int ucs_circular_buf_size(ucs_circular_buf_t *buf)
{
    return (buf->tail >= buf->head) ?
            buf->tail - buf->head :
            buf->tail + buf->num_elems - buf->head;
}

static int inline ucs_circular_buf_full(ucs_circular_buf_t *buf)
{
    return ucs_circular_buf_size(buf) == buf->num_elems - 1;
}

static int inline ucs_circular_buf_empty(ucs_circular_buf_t *buf)
{
    return ucs_circular_buf_size(buf) == 0;
}

static inline void* ucs_circular_buf_alloc(ucs_circular_buf_t *buf)
{
    if (ucs_circular_buf_full(buf)) {
        return NULL;
    }

    return UCS_PTR_BYTE_OFFSET(buf->buf, buf->elem_size * buf->tail);
}

static inline void* ucs_circular_buf_get(ucs_circular_buf_t *buf)
{
    return UCS_PTR_BYTE_OFFSET(buf->buf, buf->elem_size * buf->head);
}

static inline void
ucs_circular_buf_advance(ucs_circular_buf_t *buf, int is_head)
{
    if (is_head) {
        buf->head = (buf->head + 1) % buf->num_elems;
    } else if (!ucs_circular_buf_full(buf)) {
        buf->tail = (buf->tail + 1) % buf->num_elems;
    }
}

#endif
