/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#ifndef NVALGRIND
#  include <valgrind/memcheck.h>
#else
#  define RUNNING_ON_VALGRIND 0
#endif

#include "reloc.h"

#include <ucs/sys/compiler.h>
#include <ucm/util/sys.h>

#include <sys/fcntl.h>
#include <sys/mman.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <link.h>


typedef struct ucm_auxv {
    long               type;
    long               value;
} UCS_S_PACKED ucm_auxv_t;


typedef struct ucm_reloc_dl_iter_context {
    Dl_info            def_dlinfo;
    ucm_reloc_patch_t  *patch;
    ucs_status_t       status;
} ucm_reloc_dl_iter_context_t;


/* List of patches to be applied to additional libraries */
static UCS_LIST_HEAD(ucm_reloc_patch_list);
static void * (*ucm_reloc_orig_dlopen)(const char *, int) = NULL;
static pthread_mutex_t ucm_reloc_patch_list_lock = PTHREAD_MUTEX_INITIALIZER;


static uintptr_t
ucm_reloc_get_entry(ElfW(Addr) base, const ElfW(Phdr) *dphdr, ElfW(Sxword) tag)
{
    ElfW(Dyn) *entry;
    for (entry = (void*)(base + dphdr->p_vaddr); entry->d_tag != 0; ++entry) {
        if (entry->d_tag == tag) {
            return entry->d_un.d_val;
        }
    }
    return 0;
}

static void ucm_reloc_file_lock(int fd, int l_type)
{
    struct flock fl = { l_type, SEEK_SET, 0, 0};
    int ret;

    ret = fcntl(fd, F_SETLKW, &fl);
    if (ret < 0) {
        ucm_warn("fcntl(fd=%d, F_SETLKW, l_type=%d) failed: %m", fd, l_type);
    }
}

static int ucm_reloc_get_aux_phsize()
{
#define UCM_RELOC_AUXV_BUF_LEN 16
    static const char *proc_auxv_filename = "/proc/self/auxv";
    static int phsize = 0;
    ucm_auxv_t buffer[UCM_RELOC_AUXV_BUF_LEN];
    ucm_auxv_t *auxv;
    unsigned count;
    ssize_t nread;
    int found;
    int fd;

    /* Can avoid lock here - worst case we'll read the file more than once */
    if (phsize == 0) {
        fd = open(proc_auxv_filename, O_RDONLY);
        if (fd < 0) {
            ucm_error("failed to open '%s' for reading: %m", proc_auxv_filename);
            return fd;
        }

        if (RUNNING_ON_VALGRIND) {
            /* Work around a bug caused by valgrind's fake /proc/self/auxv -
             * every time this file is opened when running with valgrind, a
             * a duplicate of the same fd is returned, so all share the same
             * file offset.
             */
            ucm_reloc_file_lock(fd, F_WRLCK);
            lseek(fd, 0, SEEK_SET);
        }

        /* Use small buffer on the stack, avoid using malloc() */
        found = 0;
        do {
            nread = read(fd, buffer, sizeof(buffer));
            if (nread < 0) {
                ucm_error("failed to read %lu bytes from %s (ret=%ld): %m",
                          sizeof(buffer), proc_auxv_filename, nread);
                break;
            }

            count = nread / sizeof(buffer[0]);
            for (auxv = buffer; (auxv < buffer + count) && (auxv->type != AT_NULL);
                            ++auxv)
            {
                if (auxv->type == AT_PHENT) {
                    found  = 1;
                    phsize = auxv->value;
                    ucm_debug("read phent from %s: %d", proc_auxv_filename, phsize);
                    if (phsize == 0) {
                        ucm_error("phsize is 0");
                    }
                    break;
                }
            }
        } while ((count > 0) && (phsize == 0));

        if (!found) {
            ucm_error("AT_PHENT entry not found in %s", proc_auxv_filename);
        }

        if (RUNNING_ON_VALGRIND) {
            ucm_reloc_file_lock(fd, F_UNLCK);
        }
        close(fd);
    }
    return phsize;
}

ElfW(Rela) *ucm_reloc_find_sym(void *table, size_t table_size, const char *symbol,
                               void *strtab, ElfW(Sym) *symtab)
{
    ElfW(Rela) *reloc;
    char *elf_sym;

    for (reloc = table; (void*)reloc < table + table_size; ++reloc) {
        elf_sym = (char*)strtab + symtab[ELF64_R_SYM(reloc->r_info)].st_name;
        if (!strcmp(symbol, elf_sym)) {
            return reloc;
        }
    }
    return NULL;
}


static ucs_status_t
ucm_reloc_modify_got(ElfW(Addr) base, const ElfW(Phdr) *phdr, const char *phname,
                     int phnum, int phsize,
                     const ucm_reloc_dl_iter_context_t *ctx)
{
    const char *section_name;
    ElfW(Phdr) *dphdr;
    ElfW(Rela) *reloc;
    ElfW(Sym)  *symtab;
    void *jmprel, *rela, *strtab;
    size_t pltrelsz, relasz;
    long page_size;
    void **entry;
    void *prev_value;
    void *page;
    int ret;
    int i;
    Dl_info entry_dlinfo;
    int success;

    page_size = ucm_get_page_size();

    if (!strcmp(phname, "")) {
        phname = "(empty)";
    }

    /* find PT_DYNAMIC */
    dphdr = NULL;
    for (i = 0; i < phnum; ++i) {
        dphdr = (void*)phdr + phsize * i;
        if (dphdr->p_type == PT_DYNAMIC) {
            break;
        }
    }
    if (dphdr == NULL) {
        return UCS_ERR_NO_ELEM;
    }

    /* Get ELF tables pointers */
    jmprel   = (void*)ucm_reloc_get_entry(base, dphdr, DT_JMPREL);
    symtab   = (void*)ucm_reloc_get_entry(base, dphdr, DT_SYMTAB);
    strtab   = (void*)ucm_reloc_get_entry(base, dphdr, DT_STRTAB);
    pltrelsz = ucm_reloc_get_entry(base, dphdr, DT_PLTRELSZ);

    section_name = ".got.plt";
    reloc        = ucm_reloc_find_sym(jmprel, pltrelsz, ctx->patch->symbol,
                                      strtab, symtab);
    if (reloc == NULL) {
        /* if not found in .got.plt, search in .got */
        section_name = ".got";
        rela         = (void*)ucm_reloc_get_entry(base, dphdr, DT_RELA);
        relasz       = ucm_reloc_get_entry(base, dphdr, DT_RELASZ);
        reloc        = ucm_reloc_find_sym(rela, relasz, ctx->patch->symbol,
                                          strtab, symtab);
    }
    if (reloc == NULL) {
        return UCS_OK;
    }

    entry      = (void *)(base + reloc->r_offset);
    prev_value = *entry;

    if (prev_value == ctx->patch->value) {
        ucm_trace("%s entry '%s' in %s at [%p] up-to-date",
                  section_name, ctx->patch->symbol, basename(phname), entry);
        return UCS_OK;
    }

    /* enable writing to the page */
    page = (void *)((intptr_t)entry & ~(page_size - 1));
    ret  = mprotect(page, page_size, PROT_READ|PROT_WRITE);
    if (ret < 0) {
        ucm_error("failed to modify %s page %p to rw: %m", section_name, page);
        return UCS_ERR_UNSUPPORTED;
    }

    *entry = ctx->patch->value;
    ucm_debug("%s entry '%s' in %s at [%p] modified from %p to %p",
              section_name, ctx->patch->symbol, basename(phname), entry,
              prev_value, ctx->patch->value);

    success = dladdr(prev_value, &entry_dlinfo);
    ucs_assertv_always(success, "can't find shared object with symbol address %p",
                       prev_value);

    /* store default entry to prev_value to guarantee valid pointers
     * throughout life time of the process */
    if (ctx->def_dlinfo.dli_fbase == entry_dlinfo.dli_fbase) {
        ctx->patch->prev_value = prev_value;
        ucm_debug("'%s' prev_value is %p from '%s'", ctx->patch->symbol,
                  prev_value, basename(entry_dlinfo.dli_fname));
    }

    return UCS_OK;
}

static int ucm_reloc_phdr_iterator(struct dl_phdr_info *info, size_t size, void *data)
{
    ucm_reloc_dl_iter_context_t *ctx = data;
    int phsize;
    int i;

    /* check if module is black-listed for this patch */
    if (ctx->patch->blacklist) {
        for (i = 0; ctx->patch->blacklist[i]; i++) {
            if (strstr(info->dlpi_name, ctx->patch->blacklist[i])) {
                /* module is black-listed */
                ctx->status = UCS_OK;
                return 0;
            }
        }
    }

    phsize = ucm_reloc_get_aux_phsize();
    if (phsize <= 0) {
        ucm_error("failed to read phent size");
        ctx->status = UCS_ERR_UNSUPPORTED;
        return -1;
    }

    ctx->status = ucm_reloc_modify_got(info->dlpi_addr, info->dlpi_phdr,
                                       info->dlpi_name, info->dlpi_phnum,
                                       phsize, ctx);
    if (ctx->status == UCS_OK) {
        return 0; /* continue iteration and patch all objects */
    } else {
        return -1; /* stop iteration if got a real error */
    }
}

/* called with lock held */
static ucs_status_t ucm_reloc_apply_patch(ucm_reloc_patch_t *patch)
{
    ucm_reloc_dl_iter_context_t ctx;
    int                         success;

    /* Find default shared object, usually libc */
    success = dladdr(getpid, &ctx.def_dlinfo);
    if (!success) {
        return UCS_ERR_UNSUPPORTED;
    }

    ctx.patch  = patch;
    ctx.status = UCS_OK;

    /* Avoid locks here because we don't modify ELF data structures.
     * Worst case the same symbol will be written more than once.
     */
    (void)dl_iterate_phdr(ucm_reloc_phdr_iterator, &ctx);
    return ctx.status;
}

static void *ucm_dlopen(const char *filename, int flag)
{
    ucm_reloc_patch_t *patch;
    void *handle;

    if (ucm_reloc_orig_dlopen == NULL) {
        ucm_fatal("ucm_reloc_orig_dlopen is NULL");
        return NULL;
    }

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
            ucm_debug("in dlopen(%s), re-applying '%s' to %p", filename,
                      patch->symbol, patch->value);
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

/* called with lock held */
static ucs_status_t ucm_reloc_install_dlopen()
{
    static int installed = 0;
    ucs_status_t status;

    if (installed) {
        return UCS_OK;
    }

    ucm_reloc_orig_dlopen = ucm_reloc_get_orig(ucm_reloc_dlopen_patch.symbol,
                                               ucm_reloc_dlopen_patch.value);

    status = ucm_reloc_apply_patch(&ucm_reloc_dlopen_patch);
    if (status != UCS_OK) {
        return status;
    }

    ucs_list_add_tail(&ucm_reloc_patch_list, &ucm_reloc_dlopen_patch.list);

    installed = 1;
    return UCS_OK;
}

ucs_status_t ucm_reloc_modify(ucm_reloc_patch_t *patch)
{
    ucs_status_t status;

    /* Take lock first to handle a possible race where dlopen() is called
     * from another thread and we may end up not patching it.
     */
    pthread_mutex_lock(&ucm_reloc_patch_list_lock);

    status = ucm_reloc_install_dlopen();
    if (status != UCS_OK) {
        goto out_unlock;
    }

    status = ucm_reloc_apply_patch(patch);
    if (status != UCS_OK) {
        goto out_unlock;
    }

    ucs_list_add_tail(&ucm_reloc_patch_list, &patch->list);

out_unlock:
    pthread_mutex_unlock(&ucm_reloc_patch_list_lock);
    return status;
}

