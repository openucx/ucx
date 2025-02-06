/**
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef __LIB_H
#define __LIB_H
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <sys/uio.h>


typedef struct {
    void   *data;
    size_t elem_size;
    size_t count;
    size_t total;
} array_t;

void lock(void);
void unlock(void);

#define min(_a, _b) ((_a) < (_b) ? (_a) : (_b))

#define container_of(_ptr, _type, _member) \
    ((_type*)((char*)(_ptr) - (char*)&((_type*)0)->_member))

static inline void array_init(array_t *a, size_t elem_size)
{
    memset(a, 0, sizeof(*a));
    a->elem_size = elem_size;
}

static inline void array_cleanup(array_t *a)
{
    if (a->data) {
        free(a->data);
    }

    if (a->count) {
        fprintf(stderr, "ibmock: non-empty array (count=%zu)\n", a->count);
    }
}

static inline void *array_append(array_t *a, const void *data, size_t len)
{
    void *tmp, *ptr;
    size_t total;

    assert(len == a->elem_size);

    if (a->count >= a->total) {
        total = (a->total * 2)? : 8;
        tmp   = realloc(a->data, total * a->elem_size);
        if (tmp == NULL) {
            fprintf(stderr, "ibmock: OOM\n");
            exit(1);
        }

        a->data  = tmp;
        a->total = total;
    }

    ptr = a->data + (a->count * a->elem_size);
    memcpy(ptr, data, len);
    a->count++;
    return ptr;
}

static inline void *array_end(const array_t *a)
{
    return a->data + a->elem_size * a->count;
}

static inline void array_remove(array_t *a, void *data)
{
    assert(data >= a->data &&
           data + a->elem_size <= array_end(a));
    a->count--;

    if (data < array_end(a)) {
        memmove(data, data + a->elem_size, array_end(a) - data);
    }
}

#define array_foreach(_entry, _arr) \
    for (_entry = (_arr)->data; \
         (void*)(_entry + 1) <= \
         (_arr)->data + (_arr)->elem_size * (_arr)->count; \
         _entry++)


struct list {
    struct list *next, *prev;
};

static inline void list_init(struct list *head)
{
    head->next = head->prev = head;
}

static inline void list_add_tail(struct list *head, struct list *entry)
{
    head->prev->next = entry;
    entry->prev      = head->prev;
    entry->next      = head;
    head->prev       = entry;
}

static inline void list_del(struct list *entry)
{
    entry->next->prev = entry->prev;
    entry->prev->next = entry->next;
    entry->next = entry->prev = entry;
}

static inline int list_is_empty(struct list *head)
{
    return head->next == head;
}

static inline void *list_first(struct list *head)
{
    return head->next;
}

#endif /* __LIB_H */
