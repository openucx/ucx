/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifndef UCS_COMPONENT_H_
#define UCS_COMPONENT_H_

#include <ucs/sys/preprocessor.h>
#include <ucs/datastruct/list.h>
#include <ucs/type/status.h>
#include <stddef.h>

BEGIN_C_DECLS

/*
 * Component definition - used internally.
 */
typedef ucs_status_t (*ucs_component_init_cb_t)(void *base_ptr);
typedef void (*ucs_component_cleanup_cb_t)(void *base_ptr);
typedef struct ucs_component {
    ucs_component_init_cb_t     init;
    ucs_component_cleanup_cb_t  cleanup;
    size_t                      size;
    size_t                      offset;
    ucs_list_link_t             list;
} ucs_component_t;


/*
 * Helper macros
 */
#define _UCS_COMPONENT_LIST_NAME(_base_type) \
    ucs_ ## _base_type ## _component_list
#define _UCS_COMPONENT_LIST_EXTERN(_base_type) \
    extern ucs_list_link_t _UCS_COMPONENT_LIST_NAME(_base_type)


/*
 * Define code which runs at global constructor phase
 */
#define UCS_STATIC_INIT \
    static void UCS_F_CTOR UCS_PP_APPEND_UNIQUE_ID(ucs_initializer)()


/**
 * Define a list of components for specific base type.
 *
 * @param _base_type  Type to add components to.
 */
#define UCS_COMPONENT_LIST_DEFINE(_base_type) \
    UCS_LIST_HEAD(_UCS_COMPONENT_LIST_NAME(_base_type))


/**
 * Define a component for specific base type.
 *
 * @param _base_type   Type to add components to.
 * @param _name        Component name.
 * @param _init        Initialization function.
 * @param _cleanup     Cleanup function.
 * @param _size        How much room to reserve after the base type.
 */
#define UCS_COMPONENT_DEFINE(_base_type, _name, _init, _cleanup, _size) \
    \
    size_t ucs_##_name##_component_offset; \
    \
    static void UCS_F_CTOR UCS_PP_APPEND_UNIQUE_ID(ucs_component_##_name##_register)() { \
        static ucs_component_t comp = { \
            (ucs_component_init_cb_t)_init, \
            (ucs_component_cleanup_cb_t)_cleanup, \
            _size}; \
        \
        _UCS_COMPONENT_LIST_EXTERN(_base_type); \
        __ucs_component_add(&_UCS_COMPONENT_LIST_NAME(_base_type), \
                            sizeof(_base_type), &comp); \
        ucs_##_name##_component_offset = comp.offset; \
    }


/**
 * @param _base Components base type.
 * @return How much room is required for all components of this base.
 */
#define ucs_components_total_size(_base_type) \
    ({ \
        _UCS_COMPONENT_LIST_EXTERN(_base_type); \
        __ucs_components_total_size(&_UCS_COMPONENT_LIST_NAME(_base_type),\
                                    sizeof(_base_type)); \
    })


/**
 * Initialize all components of a specific base type.
 *
 * @param _base_type   Base type to initialize components for.
 * @param _base_ptr    Pointer to base type instance, to pass to components
 *                     initialization functions.
 *
 * @return UCS_OK if all components were successfully initialized, otherwise the
 *         error from the first failed component.
 */
#define ucs_components_init_all(_base_type, _base_ptr) \
    ({ \
        _UCS_COMPONENT_LIST_EXTERN(_base_type); \
        __ucs_components_init_all(&_UCS_COMPONENT_LIST_NAME(_base_type), _base_ptr); \
    })


/**
 * Cleanup all components of a specific base type.
 *
 * @param _base_type   Class whose components to cleanup.
 * @param _base_ptr    Pointer to base type instance, to pass to components
 *                     cleanup functions.
 */
#define ucs_components_cleanup_all(_base_type, _base_ptr) \
    { \
        _UCS_COMPONENT_LIST_EXTERN(_base_type); \
        __ucs_components_cleanup_all(&_UCS_COMPONENT_LIST_NAME(_base_type), _base_ptr); \
    }


/**
 * Get a component context from base type pointer..
 *
 * @param _base_ptr    Pointer to base type instance.
 * @param _type        Type of component context.
 *
 * @return Pointer to component context.
 *
 * @note: Cannot be used from library constructors.
 */
#define ucs_component_get(_base_ptr, _name, _type) \
    ({ \
        extern size_t ucs_##_name##_component_offset; \
        ((_type*)( (char*)(_base_ptr) + ucs_##_name##_component_offset )); \
    })


void __ucs_component_add(ucs_list_link_t *list, size_t base_size, ucs_component_t *comp);
size_t __ucs_components_total_size(ucs_list_link_t *list, size_t base_size);
ucs_status_t __ucs_components_init_all(ucs_list_link_t *list, void *base_ptr);
void __ucs_components_cleanup_all(ucs_list_link_t *list, void *base_ptr);

END_C_DECLS

#endif /* COMPONENT_H_ */

