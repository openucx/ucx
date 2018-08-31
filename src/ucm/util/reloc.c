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
#include <ucs/type/component.h>
#include <ucm/util/log.h>
#include <ucm/util/sys.h>

#include <sys/fcntl.h>
#include <sys/mman.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
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

static ucs_status_t
ucm_reloc_modify_got(ElfW(Addr) base, const ElfW(Phdr) *phdr, const char *phname,
                     int phnum, int phsize,
                     const ucm_reloc_dl_iter_context_t *ctx)
{
    ElfW(Phdr) *dphdr;
    ElfW(Rela) *reloc;
    ElfW(Sym)  *symtab;
    void *jmprel, *strtab;
    size_t pltrelsz;
    long page_size;
    char *elf_sym;
    void **entry;
    void *page;
    int ret;
    int i;
    Dl_info entry_dlinfo;
    int success;

    page_size = ucm_get_page_size();

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

    /* Find matching symbol and replace it */
    for (reloc = jmprel; (void*)reloc < jmprel + pltrelsz; ++reloc) {
        elf_sym = (char*)strtab + symtab[ELF64_R_SYM(reloc->r_info)].st_name;
        if (!strcmp(ctx->patch->symbol, elf_sym)) {
            entry = (void *)(base + reloc->r_offset);

            ucm_trace("'%s' entry in '%s' is at %p", ctx->patch->symbol,
                      basename(phname), entry);

            page  = (void *)((intptr_t)entry & ~(page_size - 1));
            ret = mprotect(page, page_size, PROT_READ|PROT_WRITE);
            if (ret < 0) {
                ucm_error("failed to modify GOT page %p to rw: %m", page);
                return UCS_ERR_UNSUPPORTED;
            }

            success = dladdr(*entry, &entry_dlinfo);
            ucs_assertv_always(success, "can't find shared object with entry %p",
                               *entry);

            /* store default entry to prev_value to guarantee valid pointers
             * throughout life time of the process */
            if (ctx->def_dlinfo.dli_fbase == entry_dlinfo.dli_fbase) {
                ctx->patch->prev_value = *entry;
                ucm_trace("'%s' by address %p in '%s' is stored as original for %p",
                          ctx->patch->symbol, *entry,
                          basename(entry_dlinfo.dli_fname), ctx->patch->value);
            }

            *entry = ctx->patch->value;
            break;
        }
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
    if (ctx.status == UCS_OK) {
        ucm_debug("modified '%s' from %p to %p", patch->symbol,
                  patch->prev_value, patch->value);
    }
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

void* ucm_reloc_get_orig(const char *symbol, void *replacement)
{
    const char *error;
    void *func_ptr;

    func_ptr = dlsym(RTLD_NEXT, symbol);
    if (func_ptr == NULL) {
        (void)dlerror();
        func_ptr = dlsym(RTLD_DEFAULT, symbol);
        if (func_ptr == replacement) {
            error = dlerror();
            ucm_fatal("could not find address of original %s(): %s", symbol,
                      error ? error : "Unknown error");
        }
    }

    ucm_debug("original %s() is at %p", symbol, func_ptr);
    return func_ptr;
}

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

