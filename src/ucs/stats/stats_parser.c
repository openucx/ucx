/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "stats.h"

/*
 * Dump binary statistics file to stdout.
 * Usage: ucs_stats_parser [ file1 ] [ file2 ] ...
 */

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

        ucs_stats_serialize(stdout, root, 0);
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
