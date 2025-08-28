#ifndef UCT_DEVICE_TYPES_H
#define UCT_DEVICE_TYPES_H

typedef struct uct_device_ep {

} uct_device_ep_t;

typedef uct_device_ep_t *uct_device_ep_h;

typedef void *uct_mem_element_h;

typedef struct uct_dev_completion {
} uct_dev_completion_t;

typedef enum {
    UCT_DEV_GPU_COOPERATION_THREAD,
    UCT_DEV_GPU_COOPERATION_WARP,
    UCT_DEV_GPU_COOPERATION_BLOCK,
} uct_dev_cooperation_level_t;

#endif
