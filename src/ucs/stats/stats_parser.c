/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "stats.h"
#include <inttypes.h>

/*
 * Dump binary statistics file to stdout.
 * Usage: ucs_stats_parser [ file1 ] [ file2 ] ...
 */

static ucs_status_t
dump_stats_recurs(FILE *stream, ucs_stats_node_t *node, unsigned indent)
{
    ucs_stats_node_t *child;
    unsigned i;

    fprintf(stream, "%*s" UCS_STATS_NODE_FMT ":\n", indent * 2, "",
            UCS_STATS_NODE_ARG(node));

    for (i = 0; i < node->cls->num_counters; ++i) {
        fprintf(stream, "%*s%s: %" PRIu64 "\n", (indent + 1) * 2, "",
                node->cls->counter_names[i], node->counters[i]);
    }
    ucs_list_for_each(child, &node->children[UCS_STATS_ACTIVE_CHILDREN], list) {
        dump_stats_recurs(stream, child, indent + 1);
    }

    return UCS_OK;
}

static ucs_status_t dump_file(const char *filename)
{
    ucs_stats_node_t *root;
    ucs_status_t status;
    FILE *stream;

    stream = fopen(filename, "rb");
    if (stream == NULL) {
        fprintf(stderr, "Could not open %s\n", filename);
        return UCS_ERR_IO_ERROR;
    }

    while (!feof(stream)) {
        status = ucs_stats_deserialize(stream, &root);
        if (status != UCS_OK) {
            goto out;
        }

        dump_stats_recurs(stdout, root, 0);
        ucs_stats_free(root);
    }

    status = UCS_OK;

out:
    fclose(stream);
    return status;
}

int main(int argc, char **argv)
{
    int i;

    for (i = 1; i < argc; ++i) {
        dump_file(argv[i]);
    }

    return 0;
}
