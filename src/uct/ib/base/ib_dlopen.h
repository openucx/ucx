#ifndef UCT_IB_DLOPEN_H
#define UCT_IB_DLOPEN_H

#include "ib_verbs.h"

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>

typedef struct uct_ib_dlopen_module {
    pthread_once_t once;
    void *library;
    uct_ib_dlopen_status_t status;
    const char *library_name;
    const char *missing_symbol;
    char error_msg[256];
} uct_ib_dlopen_module_t;

#define UCT_IB_DLOPEN_MODULE_INITIALIZER(_library_name) \
    {PTHREAD_ONCE_INIT, NULL, UCT_IB_DLOPEN_STATUS_OK, (_library_name), \
     NULL, {0}}

static inline void uct_ib_dlopen_module_open(uct_ib_dlopen_module_t *module)
{
    const char *dlopen_error_msg;

    module->library = dlopen(module->library_name, RTLD_NOW | RTLD_LOCAL);
    if (module->library == NULL) {
        dlopen_error_msg = dlerror();
        module->status   = UCT_IB_DLOPEN_STATUS_NO_LIB;
        snprintf(module->error_msg, sizeof(module->error_msg), "%s",
                 (dlopen_error_msg != NULL) ? dlopen_error_msg :
                 "dlopen failed");
    }
}

static inline uct_ib_dlopen_status_t
uct_ib_dlopen_module_check(const uct_ib_dlopen_module_t *module,
                           const char **library_name,
                           const char **symbol_name, const char **error_msg)
{
    if (library_name != NULL) {
        *library_name = module->library_name;
    }
    if (symbol_name != NULL) {
        *symbol_name = module->missing_symbol;
    }
    if (error_msg != NULL) {
        *error_msg = module->error_msg;
    }

    return module->status;
}

#define UCT_IB_DLOPEN_RESOLVE(_module, _ops, _field) \
    do { \
        const char *dlerror_msg; \
        dlerror(); \
        (_ops)._field = \
                (__typeof__((_ops)._field))dlsym((_module).library, #_field); \
        dlerror_msg   = dlerror(); \
        if (((_ops)._field == NULL) || (dlerror_msg != NULL)) { \
            (_module).status         = UCT_IB_DLOPEN_STATUS_MISSING_SYM; \
            (_module).missing_symbol = #_field; \
            snprintf((_module).error_msg, sizeof((_module).error_msg), "%s", \
                     (dlerror_msg != NULL) ? dlerror_msg : \
                     "symbol resolved to NULL"); \
            return; \
        } \
    } while (0)

static inline int uct_ib_dlopen_errno(uct_ib_dlopen_status_t status)
{
    switch (status) {
    case UCT_IB_DLOPEN_STATUS_OK:
        return 0;
    case UCT_IB_DLOPEN_STATUS_NO_LIB:
        return ENOENT;
    case UCT_IB_DLOPEN_STATUS_MISSING_SYM:
        return ENOSYS;
    }

    return ENOSYS;
}

#define UCT_IB_DLOPEN_OP_FIELD(_module, _ops, _ret, _fail, _name, _args, \
                               _call) \
    _ret (*_name) _args;
#define UCT_IB_DLOPEN_VOID_OP_FIELD(_module, _ops, _name, _args, _call) \
    void (*_name) _args;

#define UCT_IB_DLOPEN_RESOLVE_OP(_module, _ops, _ret, _fail, _name, _args, \
                                 _call) \
    UCT_IB_DLOPEN_RESOLVE(_module, _ops, _name);
#define UCT_IB_DLOPEN_RESOLVE_VOID_OP(_module, _ops, _name, _args, _call) \
    UCT_IB_DLOPEN_RESOLVE(_module, _ops, _name);

#define UCT_IB_DLOPEN_DEFINE_MODULE(_module, _library_name, _ops_type, \
                                    _ops_list, _check_func) \
    static uct_ib_dlopen_module_t _module##_module = \
            UCT_IB_DLOPEN_MODULE_INITIALIZER(_library_name); \
    static _ops_type _module##_ops; \
    static void _module##_init_once(void) \
    { \
        uct_ib_dlopen_module_open(&_module##_module); \
        if (_module##_module.status != UCT_IB_DLOPEN_STATUS_OK) { \
            return; \
        } \
        _ops_list(UCT_IB_DLOPEN_RESOLVE_OP, UCT_IB_DLOPEN_RESOLVE_VOID_OP, \
                  _module##_module, _module##_ops); \
    } \
    uct_ib_dlopen_status_t \
    _check_func(const char **library_name, const char **symbol_name, \
                const char **error_msg) \
    { \
        pthread_once(&_module##_module.once, _module##_init_once); \
        return uct_ib_dlopen_module_check(&_module##_module, library_name, \
                                          symbol_name, error_msg); \
    } \
    static int _module##_init(void) \
    { \
        uct_ib_dlopen_status_t status; \
        status = _check_func(NULL, NULL, NULL); \
        if (status != UCT_IB_DLOPEN_STATUS_OK) { \
            errno = uct_ib_dlopen_errno(status); \
            return -1; \
        } \
        return 0; \
    }

#define UCT_IB_DLOPEN_FWD_OP(_module, _ops, _ret, _fail, _name, _proto, _call) \
    _ret _name _proto \
    { \
        if (_module##_init() != 0) { \
            return (_fail); \
        } \
        return (_ops)._name _call; \
    }

#define UCT_IB_DLOPEN_FWD_VOID_OP(_module, _ops, _name, _proto, _call) \
    void _name _proto \
    { \
        if (_module##_init() == 0) { \
            (_ops)._name _call; \
        } \
    }

#endif
