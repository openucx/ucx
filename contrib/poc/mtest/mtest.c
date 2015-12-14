#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <link.h>
#include <elf.h>
#include <link.h>
#include <sys/mman.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>  
#include <unistd.h>
#include <errno.h>



struct strtab {
    char *tab;
    ElfW(Xword) size;
};


struct jmpreltab {
    ElfW(Rela) *tab;
    ElfW(Xword) size;
};


struct symtab {
    ElfW(Sym) *tab;
    ElfW(Xword) entsz;
};

struct auxv {
    long type;
    long value;
} __attribute__((packed));



/*************/
/* ELF stuff */
/*************/
static const ElfW(Phdr) *get_phdr_dynamic(const ElfW(Phdr) *phdr,
        uint16_t phnum, uint16_t phentsize) 
{
    int i;

    for (i = 0; i < phnum; i++) {
        if (phdr->p_type == PT_DYNAMIC)
            return phdr;
        phdr = (ElfW(Phdr) *)((char *)phdr + phentsize);
    }

    return NULL;
}



static const ElfW(Dyn) *get_dynentry(ElfW(Addr) base, const ElfW(Phdr) *pdyn,
        uint32_t type) 
{
    ElfW(Dyn) *dyn;

    for (dyn = (ElfW(Dyn) *)(base + pdyn->p_vaddr); dyn->d_tag; dyn++) {
        if (dyn->d_tag == type)
            return dyn;
    }

    return NULL;
}



static struct jmpreltab get_jmprel(ElfW(Addr) base, const ElfW(Phdr) *pdyn) 
{
    struct jmpreltab table;
    const ElfW(Dyn) *dyn;

    dyn = get_dynentry(base, pdyn, DT_JMPREL);
    table.tab = (dyn == NULL) ? NULL : (ElfW(Rela) *)dyn->d_un.d_ptr;

    dyn = get_dynentry(base, pdyn, DT_PLTRELSZ);
    table.size = (dyn == NULL) ? 0 : dyn->d_un.d_val;
    return table;
}



static struct symtab get_symtab(ElfW(Addr) base, const ElfW(Phdr) *pdyn) 
{
    struct symtab table;
    const ElfW(Dyn) *dyn;

    dyn = get_dynentry(base, pdyn, DT_SYMTAB);
    table.tab = (dyn == NULL) ? NULL : (ElfW(Sym) *)dyn->d_un.d_ptr;
    dyn = get_dynentry(base, pdyn, DT_SYMENT);
    table.entsz = (dyn == NULL) ? 0 : dyn->d_un.d_val;
    return table;
}



static struct strtab get_strtab(ElfW(Addr) base, const ElfW(Phdr) *pdyn) 
{
    struct strtab table;
    const ElfW(Dyn) *dyn;

    dyn = get_dynentry(base, pdyn, DT_STRTAB);
    table.tab = (dyn == NULL) ? NULL : (char *)dyn->d_un.d_ptr;
    dyn = get_dynentry(base, pdyn, DT_STRSZ);
    table.size = (dyn == NULL) ? 0 : dyn->d_un.d_val;
    return table;
}



static void *get_got_entry(ElfW(Addr) base, struct jmpreltab jmprel,
        struct symtab symtab, struct strtab strtab, const char *symname) 
{

    ElfW(Rela) *rela;
    ElfW(Rela) *relaend;

    relaend = (ElfW(Rela) *)((char *)jmprel.tab + jmprel.size);
    for (rela = jmprel.tab; rela < relaend; rela++) {
        uint32_t relsymidx;
        char *relsymname;
        relsymidx = ELF64_R_SYM(rela->r_info);
        relsymname = strtab.tab + symtab.tab[relsymidx].st_name;

        if (strcmp(symname, relsymname) == 0)
            return (void *)(base + rela->r_offset);
    }

    return NULL;
}


#define STRING(_s) #_s
#define conc(str1,str2) str1 ## str2 

#define REWIRE(_from) \
{\
    conc(_from,got) = get_got_entry(base, jmprel, symtab, strtab, STRING(_from));\
    if (conc(_from,got) != NULL) {\
        void *page = (void *)((intptr_t) conc(_from,got) & ~(0x1000 - 1));\
        rc = mprotect(page, 0x1000, PROT_READ | PROT_WRITE);\
        if (rc) {\
            printf("mprotect error - %s\n", strerror(errno));\
        }\
        *conc(_from,got) = conc(my,_from);\
    }\
}






int get_aux_phent()
{
    struct auxv *auxv;
    void *data = NULL;
    size_t size = 0;
    int status;
    int fd;
    static int phent = 0;

    if (phent) {
        return phent;
    }
    fd = open("/proc/self/auxv", O_RDONLY);
    if (fd < 0)
        exit(EXIT_FAILURE);

#define CHUNK_SIZE 1024

    do {
        data = realloc(data, size + CHUNK_SIZE);
        if (data == NULL)
            exit(EXIT_FAILURE);

        status = read(fd, data + size, CHUNK_SIZE);
        size += CHUNK_SIZE;
    } while (status > 0);

    for (auxv = data; auxv->type != AT_NULL; auxv++) {
        if (auxv->type == AT_PHENT) {
            /*printf("xxx AT_PHENT = %d\n", auxv->value);*/
            phent = auxv->value;
            break;
        }
    }

    (void) close(fd);
    (void) free(data);
    return phent;
}

/* avoid infinite recursive call */
static __thread int no_hook;

static void *(*mallocp)(size_t size);
static void *(*callocp)(size_t size, size_t len);
static void *(*reallocp)(void *p, size_t size);
static void *(*memalignp)(size_t size, size_t len);
static void (*freep)(void *);




/*********************************************/
/* Here come the malloc function and sisters */
/*********************************************/
static void *mymalloc(size_t len) 
{
    void *ret;
    void *caller;

    if (no_hook) {
        return (*mallocp)(len);
    }

    no_hook = 1;
    caller = (void*)(&len)[-1];
    printf("MY(%p) malloc(%zu", caller, len);
    ret = (*mallocp)(len);
    printf(") -> %p\n", ret);
    no_hook = 0;


    return ret;
}

static void *mycalloc(size_t nmemb, size_t size) 
{
    void *ret;

    if (no_hook) {
        return (*callocp)(nmemb,size);
    }

    no_hook = 1;
    printf("MY calloc(%zu,%zu", nmemb, size);
    ret = (*callocp)(nmemb,size);
    printf(") -> %p\n", ret);
    no_hook = 0;


    return ret;
}

static void myfree(void *ptr) 
{
    void *ret;

    if (no_hook) {
        return (*freep)(ptr);
    }

    no_hook = 1;
    printf("MY free(%p)\n", ptr);
    (*freep)(ptr);
    no_hook = 0;
}

static void patch_got(ElfW(Addr) base, const ElfW(Phdr) *phdr, int16_t phnum,
        int16_t phentsize) 
{

    const ElfW(Phdr) *dphdr;
    struct jmpreltab jmprel;
    struct symtab symtab;
    struct strtab strtab;

    void *(**mallocgot)(size_t);
    void  (**freegot)(void *);
    void *(**callocgot)(size_t,size_t);
    void *(**reallocgot)(void *,size_t);
    int rc;

    dphdr = get_phdr_dynamic(phdr, phnum, phentsize);
    jmprel = get_jmprel(base, dphdr);
    symtab = get_symtab(base, dphdr);
    strtab = get_strtab(base, dphdr);


    REWIRE(free);
    REWIRE(malloc);
    REWIRE(calloc);
}

static int callback(struct dl_phdr_info *info, size_t size, void *data) 
{
    uint16_t phentsize;

     /*
      *  LD_SHOW_AUXV=1 /bin/ls
      *  printf("Patching GOT entry of \"%s\"\n", info->dlpi_name);
      */
        phentsize = get_aux_phent();
        patch_got(info->dlpi_addr, info->dlpi_phdr, info->dlpi_phnum, phentsize);

    return 0;
}


__attribute__((constructor)) static void init(void) 
{
    callocp = (void *(*) (size_t, size_t)) dlsym (RTLD_NEXT, "calloc");
    mallocp = (void *(*) (size_t)) dlsym (RTLD_NEXT, "malloc");
    reallocp = (void *(*) (void *, size_t)) dlsym (RTLD_NEXT, "realloc");
    memalignp = (void *(*)(size_t, size_t)) dlsym (RTLD_NEXT, "memalign");
    freep = (void (*) (void *)) dlsym (RTLD_NEXT, "free");

    dl_iterate_phdr(callback, NULL);
}

main()
{

    void *koo = malloc(1000);
    free(koo);

    void *moo = calloc(10,1000);
    free(moo);
    printf("all done\n");
    exit(EXIT_SUCCESS);
}
