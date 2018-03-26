/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef COMPLETION_QUEUE_H_
#define COMPLETION_QUEUE_H_

#include "commands.h"
#include "context.h"

class completion_queue {
public:
    completion_queue(uint32_t cap) : queue_size(2 * cap * commands::EVENT_SIZE),
                                     event_queue(new char[queue_size]),
                                     primary_queue_offset(queue_size / 2),
                                     position(0) {}

    ~completion_queue() {
        delete[] event_queue;
    }

    void switch_primary_queue();

    char *get_event_queue() const {
        return event_queue;
    }

    uint32_t get_queue_size() const {
        return queue_size;
    }

    void add_completion(const command& cmd);

private:
    uint32_t    queue_size;
    char*       event_queue;
    uint32_t    primary_queue_offset;
    int         position;
    std::mutex  queue_lock;
};

#endif /* COMPLETION_QUEUE_H_ */
