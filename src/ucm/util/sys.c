/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "sys.h"

#include <ucm/util/log.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>


#define UCM_PROCESS_MAPS_FILE "/proc/self/maps"


size_t ucm_get_shm_seg_size(const void *shmaddr)
{
    size_t    seg_size  = 0;
    const int max_retry = 5;
    int       i         = 0;
    int       fd;

    /* NOTE: It looks like UCM_PROCESS_MAPS_FILE is updated asynchronously,
     *       then the file may be not updated after one-by-one shmat/shmdt calls
     *       from many threads
     */
    do {
        fd = open(UCM_PROCESS_MAPS_FILE, O_RDONLY);
        if (fd < 0) {
            ucm_debug("cannot open %s for reading: %m", UCM_PROCESS_MAPS_FILE);
            return 0;
        }

        seg_size = ucm_get_shm_seg_size_fd(shmaddr, fd);

        close(fd);
    } while ((seg_size == 0) && (++i < max_retry));

    return seg_size;
}

size_t ucm_get_shm_seg_size_fd(const void *shmaddr, int fd)
{
    char          buffer[1024];
    unsigned long start_addr, end_addr;
    char          *ptr, *newline;
    size_t        read_offset;
    ssize_t       nread;
    int           ret;

    read_offset = 0;
    for (;;) {
        nread = read(fd, buffer + read_offset, sizeof(buffer) - 1 - read_offset);
        if (nread < 0) {
            if (errno == EINTR) {
                continue;
            } else {
                ucm_debug("failed to read from %s: %m", UCM_PROCESS_MAPS_FILE);
                return 0;
            }
        } else if (nread == 0) {
            return 0;
        } else {
            buffer[nread + read_offset] = '\0';
        }

        ptr = buffer;
        while ( (newline = strchr(ptr, '\n')) != NULL ) {
            /* 00400000-0040b000 r-xp ... \n */
            ret = sscanf(ptr, "%lx-%lx ", &start_addr, &end_addr);
            if (ret != 2) {
                ucm_debug("Failed to parse `%s'", ptr);
                continue;
            }

            if (start_addr == (uintptr_t)shmaddr) {
                return end_addr - start_addr;
            }

            newline = strchr(ptr, '\n');
            if (newline == NULL) {
                break;
            }

            ptr = newline + 1;
        }

        read_offset = strlen(ptr);
        memmove(buffer, ptr, read_offset);
    }

    return 0;
}
