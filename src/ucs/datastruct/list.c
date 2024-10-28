/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "list.h"

#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/math.h>

void ucs_list_print_links(ucs_list_link_t *head, unsigned int num_nodes)
{
    ucs_list_link_t *node;
    unsigned int i;

    ucs_debug("List head: %p prev=%p next=%p\n", (void *)head, (void *)head->prev,
              (void *)head->next);

    node = head->next;
    for (i = 0; i < num_nodes; i++) {
        ucs_debug("Node #%u: %p prev=%p next=%p\n", i, (void *)node,
                  (void *)node->prev, (void *)node->next);
        node = node->next;
    }
}

void ucs_list_shuffle_order(ucs_list_link_t *head, unsigned int num_nodes)
{
    ucs_list_link_t **nodes_array;
    ucs_list_link_t *temp_node;
    int i, j;

    if (num_nodes <= 1) {
        ucs_debug("Nothing to shuffle, num_nodes=%u", num_nodes);
        return;
    }

    nodes_array = (ucs_list_link_t **)ucs_malloc(
        num_nodes * sizeof(*nodes_array), "nodes_array");
    if (nodes_array == NULL) {
        ucs_debug("Failed to allocate memory for nodes array");
        return;
    }

    temp_node = head->next;
    for (i = 0; i < num_nodes; i++) {
        nodes_array[i] = temp_node;
        temp_node = temp_node->next;
    }

    /* Fisher-Yates shuffle algorithm */
    for (i = num_nodes - 1; i > 0; i--) {
        j = ucs_rand() % (i + 1);
        temp_node = nodes_array[i];
        nodes_array[i] = nodes_array[j];
        nodes_array[j] = temp_node;

        if (i < num_nodes - 1) {
            nodes_array[i]->next = nodes_array[i + 1];
            nodes_array[i + 1]->prev = nodes_array[i];
        }
    }

    nodes_array[1]->prev = nodes_array[0];
    nodes_array[0]->next = nodes_array[1];
    nodes_array[0]->prev = head;
    nodes_array[num_nodes - 1]->next = head;
    head->next = nodes_array[0];
    head->prev = nodes_array[num_nodes - 1];

    ucs_free(nodes_array);
}