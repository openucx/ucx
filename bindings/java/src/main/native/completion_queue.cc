/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "completion_queue.h"

void completion_queue::add_completion(const command& cmd) {
    JUCX_LOCK(queue_lock); // Critical Section
    *(command*)(primary_queue_offset + event_queue + position) = cmd;
    position += commands::EVENT_SIZE;
}

void completion_queue::switch_primary_queue()  {
    JUCX_LOCK(queue_lock);
    primary_queue_offset = (primary_queue_offset + queue_size/2) % queue_size;
    position = 0;
}
