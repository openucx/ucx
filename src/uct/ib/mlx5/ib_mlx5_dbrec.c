/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ib_mlx5.h"

#include <ucs/sys/sys.h>
#include <ucs/arch/cpu.h>

#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>

struct uct_ib_mlx5_db_page {
    struct uct_ib_mlx5_db_page  *prev, *next;
    uint8_t                     *buf;
    int                         num_db;
    int                         use_cnt;
    struct mlx5dv_devx_umem     *mem;
    unsigned long               free[0];
};

static struct uct_ib_mlx5_db_page *uct_ib_mlx5_add_page(uct_ib_device_t *dev)
{
    uct_ib_mlx5_md_t *md = ucs_container_of(dev, uct_ib_mlx5_md_t, super.dev);
    uintptr_t ps = ucs_get_page_size();
    struct uct_ib_mlx5_db_page *page;
    int pp;
    int i;
    int nlong;
    int ret;

    pp = ps / UCS_SYS_CACHE_LINE_SIZE;
    nlong = (pp + 8 * sizeof(long) - 1) / (8 * sizeof(long));

    page = malloc(sizeof *page + nlong * sizeof(long));
    if (!page) {
        return NULL;
    }

    ret = posix_memalign((void **)&page->buf, ps, ps);
    if (ret) {
        free(page);
        return NULL;
    }

    page->num_db  = pp;
    page->use_cnt = 0;
    for (i = 0; i < nlong; ++i) {
        page->free[i] = ~0;
    }

    page->mem = mlx5dv_devx_umem_reg(dev->ibv_context, page->buf, ps,
                                     IBV_ACCESS_LOCAL_WRITE);

    page->prev = NULL;
    page->next = md->db_list;
    md->db_list = page;
    if (page->next) {
        page->next->prev = page;
    }

    return page;
}

void *uct_ib_mlx5_alloc_dbrec(uct_ib_device_t *dev, uint32_t *mem_id, size_t *off)
{
    uct_ib_mlx5_md_t *md = ucs_container_of(dev, uct_ib_mlx5_md_t, super.dev);
    void *db = NULL;
    struct uct_ib_mlx5_db_page *page;
    int i, j;

    for (page = md->db_list; page; page = page->next) {
        if (page->use_cnt < page->num_db) {
            goto found;
        }
    }

    page = uct_ib_mlx5_add_page(dev);
    if (!page) {
        goto out;
    }

found:
    ++page->use_cnt;

    for (i = 0; !page->free[i]; ++i)
        /* nothing */;

    j = ffsl(page->free[i]);
    --j;
    page->free[i] &= ~(1UL << j);

    *mem_id = page->mem->umem_id;
    *off = (i * 8 * sizeof(long) + j) * UCS_SYS_CACHE_LINE_SIZE;
    db = page->buf + *off;
out:
    return db;
}

void uct_ib_mlx5_free_dbrec(uct_ib_device_t *dev, void *db)
{
    uct_ib_mlx5_md_t *md = ucs_container_of(dev, uct_ib_mlx5_md_t, super.dev);
    uintptr_t ps = ucs_get_page_size();
    struct uct_ib_mlx5_db_page *page;
    int i;

    for (page = md->db_list; page; page = page->next) {
        if (((uintptr_t) db & ~(ps - 1)) == (uintptr_t) page->buf) {
            break;
        }
    }

    if (!page) {
        return;
    }

    i = ((uint8_t *)db - page->buf) / UCS_SYS_CACHE_LINE_SIZE;
    page->free[i / (8 * sizeof(long))] |= 1UL << (i % (8 * sizeof(long)));

    if (!--page->use_cnt) {
        if (page->prev) {
            page->prev->next = page->next;
        } else {
            md->db_list = page->next;
        }
        if (page->next) {
            page->next->prev = page->prev;
        }

        mlx5dv_devx_umem_dereg(page->mem);
        free(page->buf);
        free(page);
    }
}
