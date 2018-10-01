#include <dlfcn.h>

void* load_lib(const char *path)
{
    return dlopen(path, RTLD_NOW);
}
