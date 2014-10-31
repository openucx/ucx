/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "component.h"

#include <ucs/datastruct/list.h>


void __ucs_component_add(ucs_list_link_t *list, size_t base_size, ucs_component_t *comp)
{
    comp->offset = __ucs_components_total_size(list, base_size);
    ucs_list_add_tail(list, &comp->list);
}

size_t __ucs_components_total_size(ucs_list_link_t *list, size_t base_size)
{
    ucs_component_t *last;

    if (ucs_list_is_empty(list)) {
        return base_size;
    }

    last = ucs_list_tail(list, ucs_component_t, list);
    return last->offset + last->size;
}

void __ucs_components_cleanup(ucs_list_link_t *start, ucs_list_link_t *end, void *base_ptr)
{
    ucs_list_link_t *iter;

    iter = start;
    while (iter->next != end) {
        ucs_list_head(iter, ucs_component_t, list)->cleanup(base_ptr);
        iter = iter->next;
    }
}

ucs_status_t __ucs_components_init_all(ucs_list_link_t *list, void *base_ptr)
{
    ucs_component_t *comp;
    ucs_status_t status;

    ucs_list_for_each(comp, list, list) {
        status = comp->init(base_ptr);
        if (status != UCS_OK) {
            __ucs_components_cleanup(list, comp->list.next, base_ptr);
            return status;
        }
    }
    return UCS_OK;
}

void __ucs_components_cleanup_all(ucs_list_link_t *list, void *base_ptr)
{
    __ucs_components_cleanup(list, list, base_ptr);
}

