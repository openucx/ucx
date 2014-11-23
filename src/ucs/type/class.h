/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_OBJECT_H_
#define UCS_OBJECT_H_

#include <ucs/sys/compiler.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/type/status.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>


typedef struct ucs_class     ucs_class_t;


typedef ucs_status_t (*ucs_class_init_func_t)   (void *self, ...);
typedef void         (*ucs_class_cleanup_func_t)(void *self);

struct ucs_class {
    const char               *name;
    size_t                   size;
    ucs_class_t              *superclass;
    ucs_class_init_func_t    init;
    ucs_class_cleanup_func_t cleanup;
};


/*
 * Helper: Define names of class-related identifiers.
 */
#define _UCS_CLASS_DECL_NAME(_type) \
    UCS_PP_TOKENPASTE(_type, _class)
#define _UCS_CLASS_INIT_NAME(_type) \
    UCS_PP_TOKENPASTE(_type, _init)
#define _UCS_CLASS_CLEANUP_NAME(_type) \
    UCS_PP_TOKENPASTE(_type, _cleanup)

/**
 * Declare a class.
 *
 * @param _type     Class type.
 */
#define UCS_CLASS_DECLARE(_type) \
    extern ucs_class_t _UCS_CLASS_DECL_NAME(_type);


/**
 * Class initialization/cleanup function prototypes.
 */
#define UCS_CLASS_INIT_FUNC(_type, ...) \
    ucs_status_t _UCS_CLASS_INIT_NAME(_type)(_type *self, ## __VA_ARGS__, \
                                      ucs_class_t *_myclass, int *_init_count)
#define UCS_CLASS_CLEANUP_FUNC(_type) \
    void _UCS_CLASS_CLEANUP_NAME(_type)(_type *self)


/**
 * Define a class.
 *
 * @param _type     Class type.
 * @param _super    Superclass type (may be void to indicate top-level class)
 */
#define UCS_CLASS_DEFINE(_type, _super) \
    UCS_CLASS_DECLARE(_super) \
    ucs_class_t _UCS_CLASS_DECL_NAME(_type) = { \
        UCS_PP_QUOTE(_type), \
        sizeof(_type), \
        &_UCS_CLASS_DECL_NAME(_super), \
        (ucs_class_init_func_t)(_UCS_CLASS_INIT_NAME(_type)), \
        (ucs_class_cleanup_func_t)(_UCS_CLASS_CLEANUP_NAME(_type)) \
    };


/**
 * Initialize a class in-place.
 *
 * @param _type  Class type.
 * @param _obj   Instance pointer to initialize.
 * @param ...    Additional arguments to the constructor.
 *
 * @return UCS_OK, or error code if failed.
 */
#define UCS_CLASS_INIT(_type, _obj, ...) \
    ({ \
        UCS_CLASS_DECLARE(_type) \
        ucs_class_t *cls = &_UCS_CLASS_DECL_NAME(_type); \
        int init_count = 0; \
        ucs_status_t status; \
        \
        status = cls->init((_obj), ## __VA_ARGS__, cls, &init_count); \
        if (status != UCS_OK) { \
            _ucs_class_call_cleanup_chain(&_UCS_CLASS_DECL_NAME(_type), \
                                          (_obj), init_count); \
        } \
        \
        (status); \
    })


/**
 * Cleanup a class in-place.
 *
 * @param _type  Class type.
 * @param _obj   Instance pointer to cleanup.
 */
#define UCS_CLASS_CLEANUP(_type, _obj) \
    { \
        UCS_CLASS_DECLARE(_type) \
        _ucs_class_call_cleanup_chain(&_UCS_CLASS_DECL_NAME(_type), _obj, -1); \
    }


/**
 * Instantiate a class.
 *
 * @param _type  Class type.
 * @param _obj   Variable to save the new instance to.
 * @param ...    Additional arguments to the constructor.
 *
 * @return UCS_OK, or error code if failed.
 */
#define UCS_CLASS_NEW(_type, _obj, ...) \
    _UCS_CLASS_NEW (_type, _obj, ## __VA_ARGS__)
#define _UCS_CLASS_NEW(_type, _obj, ...) \
    ({ \
        UCS_CLASS_DECLARE(_type) \
        ucs_class_t *cls = &_UCS_CLASS_DECL_NAME(_type); \
        ucs_status_t status; \
        void *obj; \
        \
        obj = ucs_malloc(cls->size, cls->name); \
        if (obj != NULL) { \
            status = UCS_CLASS_INIT(_type, obj, ## __VA_ARGS__); \
            if (status == UCS_OK) { \
                *(_obj) = (typeof(*(_obj)))obj; /* Success - assign pointer */ \
            } else { \
                ucs_free(obj); /* Initialization failure */ \
            } \
        } else { \
            status = UCS_ERR_NO_MEMORY; /* Allocation failure */ \
        } \
        \
        (status); \
    })


/**
 * Destroy a class instance.
 *
 * @param _type  Class type.
 * @param _obj   Instance to destroy.
 */
#define UCS_CLASS_DELETE(_type, _obj) \
    { \
        UCS_CLASS_CLEANUP(_type, _obj); \
        ucs_free(_obj); \
    }


/**
 * Invoke the parent constructor.
 * Should be used only from init function (which defines "self" and "_myclass")
 *
 * @param ...     Arguments to parent constructor.
 */
#define UCS_CLASS_CALL_SUPER_INIT(...) \
    { \
        ucs_status_t status = \
            _myclass->superclass->init(self, ## __VA_ARGS__, \
                                       _myclass->superclass, _init_count); \
        if (status != UCS_OK) { \
            return status; \
        } \
        ++(*_init_count); \
    }


/**
 * Define a function which creates an instance of a class.
 *
 * @param _type     Class type.
 * @param _argtype  Type to use for the instance argument. Should be a superclass of _type.
 * @param ...       List of types for initialization arguments (without variable names).
 *
 * Defines a function which can be used as follows:
 * {
 *      ucs_status_t status;
 *      _type *obj;
 *      status = _type##_new(arg1, arg2, arg3, &obj);
 * }
 */
#define UCS_CLASS_DEFINE_NEW_FUNC(_type, _argtype, ...) \
    ucs_status_t UCS_PP_TOKENPASTE(_type, _new)( \
                    UCS_PP_FOREACH(_UCS_CLASS_INIT_ARG_DEFINE, _, \
                                   UCS_PP_ZIP((UCS_PP_SEQ(UCS_PP_NUM_ARGS(__VA_ARGS__))), (__VA_ARGS__))) \
                                   _argtype **obj_p) \
    { \
        return UCS_CLASS_NEW(_type, obj_p \
                             UCS_PP_FOREACH(_UCS_CLASS_INIT_ARG_PASS, _, \
                                            UCS_PP_SEQ(UCS_PP_NUM_ARGS(__VA_ARGS__)))); \
    }
#define _UCS_CLASS_INIT_ARG_DEFINE(_, _bundle) \
    __UCS_CLASS_INIT_ARG_DEFINE(_, UCS_PP_TUPLE_0 _bundle, UCS_PP_TUPLE_1 _bundle)
#define __UCS_CLASS_INIT_ARG_DEFINE(_, _index, _type) \
    _type _UCS_CLASS_INIT_ARG_NAME(_, _index),
#define _UCS_CLASS_INIT_ARG_PASS(_, _index) \
    , _UCS_CLASS_INIT_ARG_NAME(_, _index)
#define _UCS_CLASS_INIT_ARG_NAME(_, _index) \
    UCS_PP_TOKENPASTE(arg, _index)


/**
 * Define a function which deletes class instance.
 *
 * @param _type     Class type.
 * @param _argtype  Type to use for the instance argument. Should be a superclass of _type.
 *
 * Defines a function which can be used as follows:
 * {
 *      _type *obj = ...;
 *      _type##_delete(obj);
 */
#define UCS_CLASS_DEFINE_DELETE_FUNC(_type, _argtype) \
    void UCS_PP_TOKENPASTE(_type, _delete)(_argtype *self) \
    { \
        UCS_CLASS_DELETE(_type, self); \
    }


/**
 * Helper: Call class destructor chain.
 *
 * @param cls    Class type.
 * @param obj    Instance pointer.
 * @param limit  How many destructors to call (0: none, -1: all, 1: only ucs_object_t's).
 */
void _ucs_class_call_cleanup_chain(ucs_class_t *cls, void *obj, int limit);


#endif
