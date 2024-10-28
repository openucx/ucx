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

void ucs_list_print_nodes(const ucs_list_link_t *head, unsigned int num_nodes)
{
    ucs_list_link_t *node;
    unsigned int i;

    ucs_debug("List head: %p prev=%p next=%p\n", head, head->prev, head->next);

    node = head->next;
    for (i = 0; i < num_nodes; i++) {
        ucs_debug("Node #%u: %p prev=%p next=%p\n", i, node, node->prev,
                  node->next);
        node = node->next;
    }
}

ucs_status_t ucs_list_shuffle(ucs_list_link_t *head, unsigned int num_nodes)
{
    ucs_list_link_t **node_pointers;
    ucs_list_link_t *temp_node;
    int i, j;

    node_pointers = (ucs_list_link_t **)ucs_malloc(
        num_nodes * sizeof(*node_pointers), "node_pointers");
    if (node_pointers == NULL) {
        ucs_error("Failed to allocate memory for nodes array");
        return UCS_ERR_NO_MEMORY;
    }

    temp_node = head->next;
    for (i = 0; i < num_nodes; i++) {
        node_pointers[i] = temp_node;
        temp_node      = temp_node->next;
    }

    /* Fisher-Yates shuffle algorithm */
    for (i = num_nodes - 1; i > 0; i--) {
        j = ucs_rand() % (i + 1);

        if (i != j) {
            temp_node = node_pointers[i];
            node_pointers[i] = node_pointers[j];
            node_pointers[j] = temp_node;
        }

        if (i < num_nodes - 1) {
            node_pointers[i]->next     = node_pointers[i + 1];
            node_pointers[i + 1]->prev = node_pointers[i];
        }
    }

    node_pointers[1]->prev             = node_pointers[0];
    node_pointers[0]->next             = node_pointers[1];
    node_pointers[0]->prev             = head;
    node_pointers[num_nodes - 1]->next = head;
    head->next                         = node_pointers[0];
    head->prev                         = node_pointers[num_nodes - 1];

    ucs_free(node_pointers);
    return UCS_OK;
}
