/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "reloc.h"

#include <ucs/sys/compiler.h>
#include <ucs/type/component.h>
#include <ucm/util/log.h>
#include <ucm/util/ucm_config.h>

#include <sys/mman.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>


typedef struct ucm_elf_strtab {
    char               *tab;
    ElfW(Xword)        size;
} ucm_elf_strtab_t;

typedef struct ucm_elf_jmpreltab {
    ElfW(Rela)         *tab;
    ElfW(Xword)        size;
} ucm_elf_jmprel_t;

typedef struct ucm_elf_symtab {
    ElfW(Sym)          *tab;
    ElfW(Xword)        entsz;
} ucm_elf_symtab_t;

typedef struct ucm_auxv {
    long               type;
    long               value;
} UCS_S_PACKED ucm_auxv_t;

typedef struct ucm_reloc_dl_iter_context {
    const char         *symbol;
    void               *newvalue;
    ucs_status_t        status;
} ucm_reloc_dl_iter_context_t;


/* List of patches to be applied to additional libraries */
static UCS_LIST_HEAD(ucm_reloc_patch_list);
static void * (*ucm_reloc_orig_dlopen)(const char *, int) = NULL;
static pthread_mutex_t ucm_reloc_patch_list_lock = PTHREAD_MUTEX_INITIALIZER;


static const ElfW(Phdr) *
ucm_reloc_get_phdr_dynamic(const ElfW(Phdr) *phdr, uint16_t phnum, int phent)
{
    uint16_t i;

    for (i = 0; i < phnum; ++i) {
        if (phdr->p_type == PT_DYNAMIC) {
            return phdr;
        }
        phdr = (ElfW(Phdr)*)((char*)phdr + phent);
    }
    return NULL;
}

static const ElfW(Dyn)*
ucm_reloc_get_dynentry(ElfW(Addr) base, const ElfW(Phdr) *pdyn, uint32_t type)
{
    ElfW(Dyn) *dyn;

    for (dyn = (ElfW(Dyn)*)(base + pdyn->p_vaddr); dyn->d_tag; ++dyn) {
        if (dyn->d_tag == type) {
            return dyn;
        }
    }
    return NULL;
}

static void ucm_reloc_get_jmprel(ElfW(Addr) base, const ElfW(Phdr) *pdyn,
                                  ucm_elf_jmprel_t *table)
{
    const ElfW(Dyn) *dyn;

    dyn = ucm_reloc_get_dynentry(base, pdyn, DT_JMPREL);
    table->tab = (dyn == NULL) ? NULL : (ElfW(Rela)*)dyn->d_un.d_ptr;
    dyn = ucm_reloc_get_dynentry(base, pdyn, DT_PLTRELSZ);
    table->size = (dyn == NULL) ? 0 : dyn->d_un.d_val;
}

static void ucm_reloc_get_symtab(ElfW(Addr) base, const ElfW(Phdr) *pdyn,
                                  ucm_elf_symtab_t *table)
{
    const ElfW(Dyn) *dyn;

    dyn = ucm_reloc_get_dynentry(base, pdyn, DT_SYMTAB);
    table->tab = (dyn == NULL) ? NULL : (ElfW(Sym)*)dyn->d_un.d_ptr;
    dyn = ucm_reloc_get_dynentry(base, pdyn, DT_SYMENT);
    table->entsz = (dyn == NULL) ? 0 : dyn->d_un.d_val;
}

static void ucm_reloc_get_strtab(ElfW(Addr) base, const ElfW(Phdr) *pdyn,
                                  ucm_elf_strtab_t *table)
{
    const ElfW(Dyn) *dyn;

    dyn = ucm_reloc_get_dynentry(base, pdyn, DT_STRTAB);
    table->tab = (dyn == NULL) ? NULL : (char *)dyn->d_un.d_ptr;
    dyn = ucm_reloc_get_dynentry(base, pdyn, DT_STRSZ);
    table->size = (dyn == NULL) ? 0 : dyn->d_un.d_val;
}

static void * ucm_reloc_get_got_entry(ElfW(Addr) base, const char *symbol,
                                       const ucm_elf_jmprel_t *jmprel,
                                       const ucm_elf_symtab_t *symtab,
                                       const ucm_elf_strtab_t *strtab)
{
    ElfW(Rela) *rela, *relaend;
    const char *relsymname;
    uint32_t relsymidx;

    relaend = (ElfW(Rela) *)((char *)jmprel->tab + jmprel->size);
    for (rela = jmprel->tab; rela < relaend; ++rela) {
        relsymidx  = ELF64_R_SYM(rela->r_info);
        relsymname = strtab->tab + symtab->tab[relsymidx].st_name;
        if (!strcmp(symbol, relsymname)) {
            return (void *)(base + rela->r_offset);
        }
    }
    return NULL;
}

static int ucm_reloc_get_aux_phent()
{
    static const size_t chunk_size = 1024;
    static const char *proc_auxv_filename = "/proc/self/auxv";
    static int phent = 0;
    const ucm_auxv_t *auxv;
    ssize_t nread;
    size_t size;
    void *data;
    int fd;

    /* Can avoid lock here - worst case we'll read the file more than once */
    if (phent == 0) {
        fd = open(proc_auxv_filename, O_RDONLY);
        if (fd < 0) {
            ucm_error("failed to open '%s' for reading: %m", proc_auxv_filename);
            return fd;
        }

        data  = NULL;
        size  = 0;
        nread = 0;
        for (;;) {
            data = realloc(data, size + chunk_size);
            if (data == NULL) {
                break;
            }
            nread = read(fd, data + size, chunk_size);
            if (nread <= 0) {
                break;
            }

            size += nread;
        }
        close(fd);

        if (nread < 0) {
            free(data);
            return nread;
        } else if (data == NULL) {
            return -1;
        }

        for (auxv = data; auxv->type != AT_NULL; ++auxv) {
            if (auxv->type == AT_PHENT) {
                phent = auxv->value;
                ucm_debug("read phent from %s: %d", proc_auxv_filename, phent);
                break;
            }
        }

        free(data);
    }
    return phent;
}

static ucs_status_t
ucm_reloc_modify_got(ElfW(Addr) base, const ElfW(Phdr) *phdr, const char *phname,
                     int16_t phnum, int phent, const char *symbol, void *newvalue)
{
    const ElfW(Phdr) *dphdr;
    ucm_elf_jmprel_t jmprel;
    ucm_elf_symtab_t symtab;
    ucm_elf_strtab_t strtab;
    long page_size;
    void **entry;
    void *page;
    int ret;

    page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 0) {
        ucm_error("failed to get page size: %m");
        return UCS_ERR_IO_ERROR;
    }

    dphdr = ucm_reloc_get_phdr_dynamic(phdr, phnum, phent);

    ucm_reloc_get_jmprel(base, dphdr, &jmprel);
    ucm_reloc_get_symtab(base, dphdr, &symtab);
    ucm_reloc_get_strtab(base, dphdr, &strtab);

    entry = ucm_reloc_get_got_entry(base, symbol, &jmprel, &symtab, &strtab);
    if (entry == NULL) {
        return UCS_OK; /* Would be patched later */
    }

    page = (void *)((intptr_t)entry & ~(page_size - 1));
    ret = mprotect(page, page_size, PROT_READ|PROT_WRITE);
    if (ret < 0) {
        ucm_error("failed to modify GOT page %p to rw: %m", page);
        return UCS_ERR_UNSUPPORTED;
    }

    ucm_trace("'%s' entry in '%s' is at %p", symbol, basename(phname), entry);
    *entry    = newvalue;
    return UCS_OK;
}

static int ucm_reloc_phdr_iterator(struct dl_phdr_info *info, size_t size, void *data)
{
    ucm_reloc_dl_iter_context_t *ctx = data;
    int phent;

    phent = ucm_reloc_get_aux_phent();
    if (phent <= 0) {
        ucm_error("failed to read phent size");
        ctx->status = UCS_ERR_UNSUPPORTED;
        return -1;
    }

    ctx->status = ucm_reloc_modify_got(info->dlpi_addr, info->dlpi_phdr,
                                          info->dlpi_name, info->dlpi_phnum,
                                          phent, ctx->symbol, ctx->newvalue);
    if (ctx->status == UCS_OK) {
        return 0; /* continue iteration and patch all objects */
    } else {
        return -1; /* stop iteration if got a real error */
    }
}

static ucs_status_t ucm_reloc_apply_patch(ucm_reloc_patch_t *patch)
{
    ucm_reloc_dl_iter_context_t ctx = {
        .symbol   = patch->symbol,
        .newvalue = patch->value,
        .status   = UCS_OK
    };

    if (!ucm_global_config.enable_reloc_hooks) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* Avoid locks here because we don't modify ELF data structures.
     * Worst case the same symbol will be written more than once.
     */
    (void)dl_iterate_phdr(ucm_reloc_phdr_iterator, &ctx);
    if (ctx.status == UCS_OK) {
        ucm_debug("modified '%s' to %p", ctx.symbol, ctx.newvalue);
    }
    return ctx.status;
}

ucs_status_t ucm_reloc_modify(ucm_reloc_patch_t *patch)
{
    ucs_status_t status;

    /* Take lock first to handle a possible race where dlopen() is called
     * from another thread and we may end up not patching it.
     */
    pthread_mutex_lock(&ucm_reloc_patch_list_lock);
    status = ucm_reloc_apply_patch(patch);
    if (status == UCS_OK) {
        ucs_list_add_tail(&ucm_reloc_patch_list, &patch->list);
    }
    pthread_mutex_unlock(&ucm_reloc_patch_list_lock);
    return status;
}

static void *ucm_dlopen(const char *filename, int flag)
{
    ucm_reloc_patch_t *patch;
    void *handle;

    handle = ucm_reloc_orig_dlopen(filename, flag);
    if (handle != NULL) {
        /*
         * Every time a new object is loaded, we must update its relocations
         * with our list of patches (including dlopen itself). This code is less
         * efficient and will modify all existing objects every time, but good
         * enough.
         */
        pthread_mutex_lock(&ucm_reloc_patch_list_lock);
        ucs_list_for_each(patch, &ucm_reloc_patch_list, list) {
            ucm_debug("in dlopen(), re-applying '%s' to %p", patch->symbol,
                      patch->value);
            ucm_reloc_apply_patch(patch);
        }
        pthread_mutex_unlock(&ucm_reloc_patch_list_lock);
    }
    return handle;
}

static ucm_reloc_patch_t ucm_reloc_dlopen_patch = {
    .symbol = "dlopen",
    .value  = ucm_dlopen
};

UCS_STATIC_INIT {
    ucm_reloc_orig_dlopen = dlsym(RTLD_NEXT, ucm_reloc_dlopen_patch.symbol);
    ucm_reloc_apply_patch(&ucm_reloc_dlopen_patch);
}
