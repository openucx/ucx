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
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucm/util/sys.h>

#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <link.h>
#include <limits.h>

typedef void * (*ucm_reloc_dlopen_func_t)(const char *, int);

typedef struct ucm_auxv {
    long               type;
    long               value;
} UCS_S_PACKED ucm_auxv_t;


typedef struct ucm_reloc_dl_iter_context {
    ucm_reloc_patch_t  *patch;
    ucs_status_t       status;
    ElfW(Addr)         libucm_base_addr;  /* Base address to store previous value */
} ucm_reloc_dl_iter_context_t;


static ucm_reloc_patch_t ucm_reloc_dlopen_patch = {
    .symbol = "dlopen",
    .value  = ucm_dlopen
};


/* List of patches to be applied to additional libraries */
static UCS_LIST_HEAD(ucm_reloc_patch_list);
static ucm_reloc_dlopen_func_t ucm_reloc_orig_dlopen = NULL;
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

    for (reloc = table;
         (void*)reloc < UCS_PTR_BYTE_OFFSET(table, table_size);
         ++reloc) {
        elf_sym = (char*)strtab + symtab[ELF64_R_SYM(reloc->r_info)].st_name;
        if (!strcmp(symbol, elf_sym)) {
            return reloc;
        }
    }
    return NULL;
}


static ucs_status_t
ucm_reloc_modify_got(ElfW(Addr) base, const ElfW(Phdr) *phdr, const char UCS_V_UNUSED *phname,
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

    page_size = ucm_get_page_size();

    if (!strcmp(phname, "")) {
        phname = "(empty)";
    }

    /* find PT_DYNAMIC */
    dphdr = NULL;
    for (i = 0; i < phnum; ++i) {
        dphdr = UCS_PTR_BYTE_OFFSET(phdr, phsize * i);
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

    if ((symtab == NULL) || (strtab == NULL)) {
        /* no DT_SYMTAB or DT_STRTAB sections are defined */
        return UCS_OK;
    }

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
                  section_name, ctx->patch->symbol, ucs_basename(phname), entry);
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

    /* store default entry to prev_value to guarantee valid pointers
     * throughout life time of the process */
    if (base == ctx->libucm_base_addr) {
        ctx->patch->prev_value = prev_value;
        ucm_debug("'%s' prev_value is %p'", ctx->patch->symbol, prev_value);
    }

    return UCS_OK;
}

static int ucm_reloc_phdr_iterator(struct dl_phdr_info *info, size_t size,
                                   void *data)
{
    ucm_reloc_dl_iter_context_t *ctx = data;
    int phsize;
    int i;

    /* check if shared object is black-listed for this patch */
    if (ctx->patch->blacklist) {
        for (i = 0; ctx->patch->blacklist[i]; i++) {
            if (strstr(info->dlpi_name, ctx->patch->blacklist[i])) {
                /* shared object is black-listed */
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
    if (ctx->status != UCS_OK) {
        return -1; /* stop iteration if got a real error */
    }

    /* Continue iteration and patch all remaining objects. */
    return 0;
}

/* called with lock held */
static ucs_status_t ucm_reloc_apply_patch(ucm_reloc_patch_t *patch,
                                          ElfW(Addr) libucm_base_addr)
{
    ucm_reloc_dl_iter_context_t ctx;

    ctx.patch              = patch;
    ctx.status             = UCS_OK;
    ctx.libucm_base_addr   = libucm_base_addr;

    /* Avoid locks here because we don't modify ELF data structures.
     * Worst case the same symbol will be written more than once.
     */
    (void)dl_iterate_phdr(ucm_reloc_phdr_iterator, &ctx);
    return ctx.status;
}

/* read serinfo from 'module_path', result buffer must be destroyed
 * by free() call */
static Dl_serinfo *ucm_dlopen_load_serinfo(const char *module_path)
{
    Dl_serinfo *serinfo = NULL;
    Dl_serinfo serinfo_size;
    void *module;
    int res;

    module = ucm_reloc_orig_dlopen(module_path, RTLD_LAZY);
    if (module == NULL) { /* requested module can't be loaded */
        ucm_debug("failed to open %s: %s", module_path, dlerror());
        return NULL;
    }

    /* try to get search info from requested module */
    res = dlinfo(module, RTLD_DI_SERINFOSIZE, &serinfo_size);
    if (res) {
        ucm_debug("dlinfo(RTLD_DI_SERINFOSIZE) failed");
        goto close_module;
    }

    serinfo = malloc(serinfo_size.dls_size);
    if (serinfo == NULL) {
        ucm_error("failed to allocate %zu bytes for Dl_serinfo",
                  serinfo_size.dls_size);
        goto close_module;
    }

    *serinfo = serinfo_size;
    res      = dlinfo(module, RTLD_DI_SERINFO, serinfo);
    if (res) {
        ucm_debug("dlinfo(RTLD_DI_SERINFO) failed");
        free(serinfo);
        serinfo = NULL;
    }

close_module:
    dlclose(module);
    return serinfo;
}

void *ucm_dlopen(const char *filename, int flag)
{
    void *handle;
    ucm_reloc_patch_t *patch;
    Dl_serinfo *serinfo;
    Dl_info dl_info;
    int res;
    int i;
    char file_path[PATH_MAX];
    struct stat file_stat;

    ucm_debug("open module: %s, flag: %x", filename, flag);

    if (ucm_reloc_orig_dlopen == NULL) {
        ucm_reloc_orig_dlopen =
            (ucm_reloc_dlopen_func_t)ucm_reloc_get_orig(ucm_reloc_dlopen_patch.symbol,
                                                        ucm_reloc_dlopen_patch.value);

        if (ucm_reloc_orig_dlopen == NULL) {
            ucm_fatal("ucm_reloc_orig_dlopen is NULL");
        }
    }

    if (!ucm_global_opts.dlopen_process_rpath) {
        goto fallback_load_lib;
    }

    if (filename == NULL) {
        /* return handle to main program */
        goto fallback_load_lib;
    }

    /* failed to open module directly, try to use RPATH from from caller
     * to locate requested module */
    if (filename[0] == '/') { /* absolute path - fallback to legacy mode */
        goto fallback_load_lib;
    }

    /* try to get module info */
    res = dladdr(__builtin_return_address(0), &dl_info);
    if (!res) {
        ucm_debug("dladdr failed");
        goto fallback_load_lib;
    }

    serinfo = ucm_dlopen_load_serinfo(dl_info.dli_fname);
    if (serinfo == NULL) {
        /* failed to load serinfo, try just dlopen */
        goto fallback_load_lib;
    }

    for (i = 0; i < serinfo->dls_cnt; i++) {
        ucm_concat_path(file_path, sizeof(file_path),
                        serinfo->dls_serpath[i].dls_name, filename);
        ucm_debug("check for %s", file_path);

        res = stat(file_path, &file_stat);
        if (res) {
            continue;
        }

        free(serinfo);
        /* ok, file exists, let's try to load it */
        handle = ucm_reloc_orig_dlopen(file_path, flag);
        if (handle == NULL) {
            return NULL;
        }

        goto out_apply_patches;
    }

    free(serinfo);
    /* ok, we can't lookup module in dirs listed in caller module,
     * let's fallback to legacy mode */
fallback_load_lib:
    handle = ucm_reloc_orig_dlopen(filename, flag);
    if (handle == NULL) {
        return NULL;
    }

out_apply_patches:
    /*
     * Every time a new shared object is loaded, we must update its relocations
     * with our list of patches (including dlopen itself). We have to go over
     * the entire list of shared objects, since there more objects could be
     * loaded due to dependencies.
     */

    pthread_mutex_lock(&ucm_reloc_patch_list_lock);
    ucs_list_for_each(patch, &ucm_reloc_patch_list, list) {
        ucm_debug("in dlopen(%s), re-applying '%s' to %p", filename,
                  patch->symbol, patch->value);
        ucm_reloc_apply_patch(patch, 0);
    }
    pthread_mutex_unlock(&ucm_reloc_patch_list_lock);

    return handle;
}

/* called with lock held */
static ucs_status_t ucm_reloc_install_dlopen()
{
    static int installed = 0;
    ucs_status_t status;

    if (installed) {
        return UCS_OK;
    }

    ucm_reloc_orig_dlopen =
        (ucm_reloc_dlopen_func_t)ucm_reloc_get_orig(ucm_reloc_dlopen_patch.symbol,
                                                    ucm_reloc_dlopen_patch.value);

    status = ucm_reloc_apply_patch(&ucm_reloc_dlopen_patch, 0);
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
    Dl_info dlinfo;
    int ret;

    /* Take default symbol value from the current library */
    ret = dladdr(ucm_reloc_modify, &dlinfo);
    if (!ret) {
        ucm_error("dladdr() failed to query current library");
        return UCS_ERR_UNSUPPORTED;
    }

    /* Take lock first to handle a possible race where dlopen() is called
     * from another thread and we may end up not patching it.
     */
    pthread_mutex_lock(&ucm_reloc_patch_list_lock);

    status = ucm_reloc_install_dlopen();
    if (status != UCS_OK) {
        goto out_unlock;
    }

    status = ucm_reloc_apply_patch(patch, (uintptr_t)dlinfo.dli_fbase);
    if (status != UCS_OK) {
        goto out_unlock;
    }

    ucs_list_add_tail(&ucm_reloc_patch_list, &patch->list);

out_unlock:
    pthread_mutex_unlock(&ucm_reloc_patch_list_lock);
    return status;
}

