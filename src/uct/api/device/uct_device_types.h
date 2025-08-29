#ifndef UCT_DEVICE_TYPES_H
#define UCT_DEVICE_TYPES_H

typedef struct uct_device_ep {
    unsigned uct_tl_id;
} uct_device_ep_t;

typedef uct_device_ep_t *uct_device_ep_h;

typedef void *uct_device_mem_element_h;

typedef struct uct_device_completion {
    unsigned count;
} uct_device_completion_t;

typedef enum {
    UCT_DEV_GPU_COOPERATION_LEVEL_THREAD,
    UCT_DEV_GPU_COOPERATION_LEVEL_WARP,
    UCT_DEV_GPU_COOPERATION_LEVEL_BLOCK
} uct_device_cooperation_level_t;

#endif
