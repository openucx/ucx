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

#include <ucm/util/khash_safe.h>
#include <ucm/util/sys.h>

#include <ucs/sys/compiler.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>

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
typedef int    (*ucm_reloc_dlclose_func_t)(void *);

typedef struct ucm_auxv {
    long               type;
    long               value;
} UCS_S_PACKED ucm_auxv_t;


typedef struct ucm_reloc_dl_iter_context {
    ucm_reloc_patch_t  *patch;
    ucs_status_t       status;
    ElfW(Addr)         libucm_base_addr;  /* Base address to store previous value */
} ucm_reloc_dl_iter_context_t;


/* Hash of symbols in a dynamic object */
KHASH_MAP_INIT_STR(ucm_dl_symbol_hash, void*);

/* Hash of loaded dynamic objects */
typedef struct {
    khash_t(ucm_dl_symbol_hash) symbols;
    uintptr_t                   start, end;
} ucm_dl_info_t;

KHASH_MAP_INIT_INT64(ucm_dl_info_hash, ucm_dl_info_t)

/* List of patches to be applied to additional libraries */
static UCS_LIST_HEAD(ucm_reloc_patch_list);
static pthread_mutex_t ucm_reloc_patch_list_lock = PTHREAD_MUTEX_INITIALIZER;

static khash_t(ucm_dl_info_hash) ucm_dl_info_hash;
static ucm_reloc_dlopen_func_t  ucm_reloc_orig_dlopen  = NULL;
static ucm_reloc_dlclose_func_t ucm_reloc_orig_dlclose = NULL;

/* forward declaration */
static void ucm_reloc_get_orig_dl_funcs();

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

static ucs_status_t ucm_reloc_get_aux_phsize(int *phsize_p)
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
    if (phsize != 0) {
        *phsize_p = phsize;
        return UCS_OK;
    }

    fd = open(proc_auxv_filename, O_RDONLY);
    if (fd < 0) {
        ucm_error("failed to open '%s' for reading: %m", proc_auxv_filename);
        return UCS_ERR_IO_ERROR;
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
        for (auxv = buffer; (auxv < (buffer + count)) && (auxv->type != AT_NULL);
             ++auxv)
        {
            if ((auxv->type == AT_PHENT) && (auxv->value > 0)) {
                found  = 1;
                phsize = auxv->value;
                ucm_debug("read phent from %s: %d", proc_auxv_filename, phsize);
                if (phsize == 0) {
                    ucm_error("phsize is 0");
                }
                break;
            }
        }
    } while ((count > 0) && !found);

    if (RUNNING_ON_VALGRIND) {
        ucm_reloc_file_lock(fd, F_UNLCK);
    }
    close(fd);

    if (!found) {
        ucm_error("AT_PHENT entry not found in %s", proc_auxv_filename);
        return UCS_ERR_NO_ELEM;
    }

    *phsize_p = phsize;
    return UCS_OK;
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
ucm_reloc_dl_apply_patch(const ucm_dl_info_t *dl_info, const char *dl_basename,
                         int store_prev, ucm_reloc_patch_t *patch)
{
    void *prev_value;
    khiter_t khiter;
    long page_size;
    void **entry;
    void *page;
    int ret;

    /* find symbol in our hash table */
    khiter = kh_get(ucm_dl_symbol_hash, &dl_info->symbols, patch->symbol);
    if (khiter == kh_end(&dl_info->symbols)) {
        ucm_trace("symbol '%s' not found in %s", patch->symbol, dl_basename);
        return UCS_OK;
    }

    /* get entry address from hash table */
    entry      = kh_val(&dl_info->symbols, khiter);
    prev_value = *entry;

    if (prev_value == patch->value) {
        ucm_trace("symbol '%s' in %s at [%p] up-to-date", patch->symbol,
                  dl_basename, entry);
        return UCS_OK;
    }

    /* enable writing to the page */
    page_size = ucm_get_page_size();
    page      = ucs_align_down_pow2_ptr(entry, page_size);
    ret       = mprotect(page, page_size, PROT_READ|PROT_WRITE);
    if (ret < 0) {
        ucm_error("failed to modify %s page %p to rw: %m", dl_basename, page);
        return UCS_ERR_UNSUPPORTED;
    }

    /* modify the relocation to the new value */
    *entry = patch->value;
    ucm_debug("symbol '%s' in %s at [%p] modified from %p to %p",
              patch->symbol, dl_basename, entry, prev_value, patch->value);

    /* store default entry to prev_value to guarantee valid pointers
     * throughout life time of the process
     * ignore symbols which point back to the same library, since they probably
     * point to own .plt rather than to the real function.
     */
    if (store_prev &&
        !((prev_value >= (void*)dl_info->start) &&
          (prev_value <  (void*)dl_info->end))) {
        patch->prev_value = prev_value;
        ucm_debug("'%s' prev_value is %p", patch->symbol, prev_value);
    }

    return UCS_OK;
}

static unsigned
ucm_dl_populate_symbols(ucm_dl_info_t *dl_info, uintptr_t dlpi_addr, void *table,
                        size_t table_size, void *strtab, ElfW(Sym) *symtab,
                        const char *dl_name)
{
    ElfW(Rela) *reloc;
    khiter_t khiter;
    unsigned count;
    char *elf_sym;
    int ret;

    count = 0;
    for (reloc = table; (void*)reloc < UCS_PTR_BYTE_OFFSET(table, table_size);
         ++reloc) {
        elf_sym = (char*)strtab + symtab[ELF64_R_SYM(reloc->r_info)].st_name;
        if (*elf_sym == '\0') {
            /* skip empty symbols */
            continue;
        }

        khiter = kh_put(ucm_dl_symbol_hash, &dl_info->symbols, elf_sym, &ret);
        if ((ret == UCS_KH_PUT_BUCKET_EMPTY) ||
            (ret == UCS_KH_PUT_BUCKET_CLEAR)) {
            /* do not override previous values */
            kh_val(&dl_info->symbols, khiter) = (void*)(dlpi_addr +
                                                        reloc->r_offset);
            ++count;
        } else if (ret == UCS_KH_PUT_KEY_PRESENT) {
            ucm_trace("ignoring duplicate symbol '%s' in %s", elf_sym, dl_name);
        } else {
            ucm_debug("failed to add symbol '%s' in %s", elf_sym, dl_name);
        }
    }

    return count;
}

static ucs_status_t ucm_reloc_dl_info_get(const struct dl_phdr_info *phdr_info,
                                          const char *dl_name,
                                          const ucm_dl_info_t **dl_info_p)
{
    uintptr_t dlpi_addr = phdr_info->dlpi_addr;
    unsigned UCS_V_UNUSED num_symbols;
    void *jmprel, *rela, *strtab;
    size_t pltrelsz, relasz;
    ucm_dl_info_t *dl_info;
    ucs_status_t status;
    ElfW(Phdr) *phdr, *dphdr;
    int i, ret, found_pt_load;
    ElfW(Sym) *symtab;
    khiter_t khiter;
    int phsize;

    status = ucm_reloc_get_aux_phsize(&phsize);
    if (status != UCS_OK) {
        return status;
    }

    khiter = kh_put(ucm_dl_info_hash, &ucm_dl_info_hash, dlpi_addr, &ret);
    if (ret == UCS_KH_PUT_FAILED) {
        ucm_error("failed to add dl info hash entry");
        return UCS_ERR_NO_MEMORY;
    }

    dl_info = &kh_val(&ucm_dl_info_hash, khiter);
    if (ret == UCS_KH_PUT_KEY_PRESENT) {
        /* exists */
        goto out;
    }

    kh_init_inplace(ucm_dl_symbol_hash, &dl_info->symbols);
    dl_info->start = UINTPTR_MAX;
    dl_info->end   = 0;

    /* Scan program headers for PT_LOAD and PT_DYNAMIC */
    dphdr         = NULL;
    found_pt_load = 0;
    for (i = 0; i < phdr_info->dlpi_phnum; ++i) {
        phdr = UCS_PTR_BYTE_OFFSET(phdr_info->dlpi_phdr, phsize * i);
        if (phdr->p_type == PT_LOAD) {
            /* Found loadable section - update address range */
            dl_info->start = ucs_min(dl_info->start, dlpi_addr + phdr->p_vaddr);
            dl_info->end   = ucs_max(dl_info->end, phdr->p_vaddr + phdr->p_memsz);
            found_pt_load  = 1;
        } else if (phdr->p_type == PT_DYNAMIC) {
            /* Found dynamic section */
            dphdr = phdr;
        }
    }

    if (dphdr == NULL) {
        ucm_debug("%s has no dynamic section - skipping", dl_name)
        goto out;
    }

    if (!found_pt_load) {
        ucm_debug("%s has no loaded sections - skipping", dl_name)
        goto out;
    }

    /* Get ELF tables pointers */
    symtab = (void*)ucm_reloc_get_entry(dlpi_addr, dphdr, DT_SYMTAB);
    strtab = (void*)ucm_reloc_get_entry(dlpi_addr, dphdr, DT_STRTAB);
    if ((symtab == NULL) || (strtab == NULL)) {
        /* no DT_SYMTAB or DT_STRTAB sections are defined */
        ucm_debug("%s has no dynamic symbols - skipping", dl_name)
        goto out;
    }

    num_symbols = 0;

    /* populate .got.plt */
    jmprel = (void*)ucm_reloc_get_entry(dlpi_addr, dphdr, DT_JMPREL);
    if (jmprel != NULL) {
        pltrelsz     = ucm_reloc_get_entry(dlpi_addr, dphdr, DT_PLTRELSZ);
        num_symbols += ucm_dl_populate_symbols(dl_info, dlpi_addr, jmprel,
                                               pltrelsz, strtab, symtab, dl_name);
    }

    /* populate .got */
    rela = (void*)ucm_reloc_get_entry(dlpi_addr, dphdr, DT_RELA);
    if (rela != NULL) {
        relasz       = ucm_reloc_get_entry(dlpi_addr, dphdr, DT_RELASZ);
        num_symbols += ucm_dl_populate_symbols(dl_info, dlpi_addr, rela, relasz,
                                               strtab, symtab, dl_name);
    }

    ucm_debug("added dl_info %p for %s with %u symbols range 0x%lx..0x%lx",
              dl_info, ucs_basename(dl_name), num_symbols, dl_info->start,
              dl_info->end);

out:
    *dl_info_p = dl_info;
    return UCS_OK;
}

static void ucm_reloc_dl_info_cleanup(ElfW(Addr) dlpi_addr, const char *dl_name)
{
    ucm_dl_info_t *dl_info;
    khiter_t khiter;

    khiter = kh_get(ucm_dl_info_hash, &ucm_dl_info_hash, dlpi_addr);
    if (khiter == kh_end(&ucm_dl_info_hash)) {
        ucm_debug("no dl_info entry for address 0x%lx", dlpi_addr);
        return;
    }

    /* destroy symbols hash table */
    dl_info = &kh_val(&ucm_dl_info_hash, khiter);
    kh_destroy_inplace(ucm_dl_symbol_hash, &dl_info->symbols);

    /* delete entry in dl_info hash */
    kh_del(ucm_dl_info_hash, &ucm_dl_info_hash, khiter);

    ucm_debug("removed dl_info %p for %s", dl_info, ucs_basename(dl_name));
}

static int
ucm_reloc_patch_is_dl_blacklisted(const char *dlpi_name,
                                  const ucm_reloc_patch_t *patch)
{
    unsigned i;

    if (patch->blacklist == NULL) {
        return 0;
    }

    for (i = 0; patch->blacklist[i] != NULL; i++) {
        if (strstr(dlpi_name, patch->blacklist[i])) {
            return 1;
        }
    }

    return 0;
}

static const char*
ucm_reloc_get_dl_name(const char *dlpi_name, ElfW(Addr) dlpi_addr, char *buf,
                      size_t max)
{
    if (strcmp(dlpi_name, "")) {
        return dlpi_name;
    } else {
        snprintf(buf, max, "(anonymous dl @ 0x%lx)", dlpi_addr);
        return buf;
    }
}

static int ucm_reloc_phdr_iterator(struct dl_phdr_info *phdr_info, size_t size,
                                   void *data)
{
    ucm_reloc_dl_iter_context_t *ctx = data;
    const ucm_dl_info_t *dl_info;
    char dl_name_buffer[256];
    const char *dl_name;
    int store_prev;

    /* check if shared object is black-listed for this patch */
    if (ucm_reloc_patch_is_dl_blacklisted(phdr_info->dlpi_name, ctx->patch)) {
        return 0;
    }

    dl_name = ucm_reloc_get_dl_name(phdr_info->dlpi_name, phdr_info->dlpi_addr,
                                    dl_name_buffer, sizeof(dl_name_buffer));

    ctx->status = ucm_reloc_dl_info_get(phdr_info, dl_name, &dl_info);
    if (ctx->status != UCS_OK) {
        return -1; /* stop iteration if got a real error */
    }

    /*
     * Prefer taking the previous value from main program, if exists,
     * Otherwise, use the current module (libucm.so)
     */
    store_prev = (phdr_info->dlpi_addr == 0) ||
                 ((ctx->patch->prev_value == NULL) &&
                  (phdr_info->dlpi_addr == ctx->libucm_base_addr));

    ctx->status = ucm_reloc_dl_apply_patch(dl_info, ucs_basename(dl_name),
                                           store_prev, ctx->patch);
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
    ucm_trace("patch symbol '%s'", patch->symbol);
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
    ucm_reloc_orig_dlclose(module);
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

    ucm_reloc_get_orig_dl_funcs();

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

    ucm_trace("dlopen(%s) = %p", filename, handle);

    pthread_mutex_lock(&ucm_reloc_patch_list_lock);
    ucs_list_for_each(patch, &ucm_reloc_patch_list, list) {
        ucm_debug("in dlopen(%s), re-applying '%s' to %p", filename,
                  patch->symbol, patch->value);
        ucm_reloc_apply_patch(patch, 0);
    }
    pthread_mutex_unlock(&ucm_reloc_patch_list_lock);

    return handle;
}

static int ucm_dlclose(void *handle)
{
    struct link_map *lm_entry;
    char dl_name_buffer[256];
    const char *dl_name;
    int ret;

    ucm_trace("dlclose(%p)", handle);

    ret = dlinfo(handle, RTLD_DI_LINKMAP, &lm_entry);
    if (ret != 0) {
        ucm_warn("dlinfo(handle=%p) failed during dlclose() hook, symbol"
                 "table may become unreliable", handle);
    } else {
        /*
         * Cleanup the cached information about the library.
         * NOTE: The library may not actually be unloaded (if its reference
         * count is > 1). Since we have no safe way to know it, we remove the
         * cached information anyway, and it may be re-added on the next call to
         * ucm_reloc_apply_patch().
         */
        dl_name = ucm_reloc_get_dl_name(lm_entry->l_name, lm_entry->l_addr,
                                        dl_name_buffer, sizeof(dl_name_buffer));
        pthread_mutex_lock(&ucm_reloc_patch_list_lock);
        ucm_reloc_dl_info_cleanup(lm_entry->l_addr, dl_name);
        pthread_mutex_unlock(&ucm_reloc_patch_list_lock);
    }

    ucm_reloc_get_orig_dl_funcs();

    return ucm_reloc_orig_dlclose(handle);
}

static ucm_reloc_patch_t ucm_dlopen_reloc_patches[] = {
    { .symbol = "dlopen",  .value  = ucm_dlopen  },
    { .symbol = "dlclose", .value  = ucm_dlclose }
};

static void ucm_reloc_get_orig_dl_funcs()
{
    ucm_reloc_patch_t *patch;

    /* pointer to original dlopen() */
    if (ucm_reloc_orig_dlopen == NULL) {
        patch                 = &ucm_dlopen_reloc_patches[0];
        ucm_reloc_orig_dlopen = (ucm_reloc_dlopen_func_t)
                                ucm_reloc_get_orig(patch->symbol, patch->value);
        if (ucm_reloc_orig_dlopen == NULL) {
            ucm_fatal("ucm_reloc_orig_dlopen is NULL");
        }
    }

    /* pointer to original dlclose() */
    if (ucm_reloc_orig_dlclose == NULL) {
        patch                  = &ucm_dlopen_reloc_patches[1];
        ucm_reloc_orig_dlclose = (ucm_reloc_dlclose_func_t)
                                 ucm_reloc_get_orig(patch->symbol, patch->value);
        if (ucm_reloc_orig_dlclose == NULL) {
            ucm_fatal("ucm_reloc_orig_dlclose is NULL");
        }
    }
}

/* called with lock held */
static ucs_status_t ucm_reloc_install_dl_hooks()
{
    static int installed = 0;
    ucs_status_t status;
    size_t i;

    if (installed) {
        return UCS_OK;
    }

    for (i = 0; i < ucs_static_array_size(ucm_dlopen_reloc_patches); ++i) {
        status = ucm_reloc_apply_patch(&ucm_dlopen_reloc_patches[i], 0);
        if (status != UCS_OK) {
            return status;
        }

        ucs_list_add_tail(&ucm_reloc_patch_list, &ucm_dlopen_reloc_patches[i].list);
    }

    installed = 1;
    return UCS_OK;
}

ucs_status_t ucm_reloc_modify(ucm_reloc_patch_t *patch)
{
    ucs_status_t status;
    Dl_info dl_info;
    int ret;

    ucm_reloc_get_orig_dl_funcs();

    /* Take default symbol value from the current library */
    ret = dladdr(ucm_reloc_modify, &dl_info);
    if (!ret) {
        ucm_error("dladdr() failed to query current library");
        return UCS_ERR_UNSUPPORTED;
    }

    /* Take lock first to handle a possible race where dlopen() is called
     * from another thread and we may end up not patching it.
     */
    pthread_mutex_lock(&ucm_reloc_patch_list_lock);

    status = ucm_reloc_install_dl_hooks();
    if (status != UCS_OK) {
        goto out_unlock;
    }

    status = ucm_reloc_apply_patch(patch, (uintptr_t)dl_info.dli_fbase);
    if (status != UCS_OK) {
        goto out_unlock;
    }

    ucs_list_add_tail(&ucm_reloc_patch_list, &patch->list);

out_unlock:
    pthread_mutex_unlock(&ucm_reloc_patch_list_lock);
    return status;
}

UCS_STATIC_INIT {
    kh_init_inplace(ucm_dl_info_hash, &ucm_dl_info_hash);
}
