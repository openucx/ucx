#ifndef UCT_DEVICE_TYPES_H
#define UCT_DEVICE_TYPES_H

#define UCT_DEVICE_FUNC(_ret_type, _name, ...) \
    UCS_F_DEVICE _ret_type _name(__VA_ARGS__)
#define UCT_DEVICE_FUNC_VOID(_name, ...) \
    UCS_F_DEVICE void _name(__VA_ARGS__)

typedef void *uct_mem_element_h;

typedef struct uct_dev_completion {
} uct_dev_completion_t;

typedef enum {
    UCT_DEV_SCALE_THREAD,
    UCT_DEV_SCALE_WARP,
    UCT_DEV_SCALE_BLOCK,
    /* TODO add
    UCT_DEV_SCALE_GRID */
} uct_dev_scale_t;

#endif
