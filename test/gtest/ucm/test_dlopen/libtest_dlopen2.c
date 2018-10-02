#include <sys/mman.h>
#include <errno.h>

void fire_mmap(void)
{
    void* map_ptr;

    map_ptr = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    munmap(map_ptr, 4096);
}
