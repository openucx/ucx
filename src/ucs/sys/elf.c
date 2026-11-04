/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "elf.h"

#include <ucs/datastruct/array.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/ptr_arith.h>

#include <elf.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>


/**
 * Validate ELF64 header and section header table, then resolve .shstrtab.
 * On success, sets *shdr_p, *shstrtab_p, *shstrtab_size_p for use by the caller.
 */
static ucs_status_t ucs_elf_get_section_table(const void *base, size_t file_size,
                                              const Elf64_Shdr **shdr_p,
                                              const char **shstrtab_p,
                                              Elf64_Word *shstrtab_size_p)
{
    const Elf64_Ehdr *ehdr = base;
    const Elf64_Shdr *shdr;
    const char *shstrtab;
    size_t shdr_end;
    size_t shstrtab_end;

    if (file_size < sizeof(Elf64_Ehdr)) {
        return UCS_ERR_IO_ERROR;
    }

    shdr_end = ehdr->e_shoff + ehdr->e_shnum * sizeof(Elf64_Shdr);
    if (shdr_end > file_size) {
        ucs_warn("invalid ELF: section header table out of bounds");
        return UCS_ERR_IO_ERROR;
    }

    if (ehdr->e_shstrndx >= ehdr->e_shnum) {
        ucs_warn("invalid ELF: bad e_shstrndx");
        return UCS_ERR_IO_ERROR;
    }

    shdr     = UCS_PTR_BYTE_OFFSET(base, ehdr->e_shoff);
    shstrtab = UCS_PTR_BYTE_OFFSET(base, shdr[ehdr->e_shstrndx].sh_offset);
    shstrtab_end = shdr[ehdr->e_shstrndx].sh_offset + shdr[ehdr->e_shstrndx].sh_size;

    if (shstrtab_end > file_size) {
        ucs_warn("invalid ELF: .shstrtab out of bounds");
        return UCS_ERR_IO_ERROR;
    }

    *shdr_p          = shdr;
    *shstrtab_p      = shstrtab;
    *shstrtab_size_p = shdr[ehdr->e_shstrndx].sh_size;

    return UCS_OK;
}

static int ucs_elf_section_matches_prefix(const char *shstrtab,
                                          Elf64_Word shstrndx_size,
                                          Elf64_Word sh_name,
                                          const char *name_prefix)
{
    size_t prefix_len = strlen(name_prefix);

    if ((sh_name >= shstrndx_size) || ((sh_name + prefix_len) > shstrndx_size)) {
        return 0;
    }
    return strncmp(UCS_PTR_BYTE_OFFSET(shstrtab, sh_name),
                   name_prefix, prefix_len) == 0;
}

static ucs_status_t ucs_elf_parse_note(const void *base, size_t file_size,
                                      const Elf64_Shdr *shdr,
                                      const char *sec_name,
                                      ucs_elf_note_t *note_out)
{
    const Elf32_Nhdr *note_header;
    size_t name_align;
    size_t desc_align;
    size_t note_size;
    const char *name;
    const void *desc;
    ucs_status_t status;

    if (shdr->sh_offset + sizeof(Elf32_Nhdr) > file_size) {
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    note_header = UCS_PTR_BYTE_OFFSET(base, shdr->sh_offset);
    name_align  = ucs_align_up_pow2(note_header->n_namesz, sizeof(Elf32_Word));
    desc_align  = ucs_align_up_pow2(note_header->n_descsz, sizeof(Elf32_Word));
    note_size   = sizeof(Elf32_Nhdr) + name_align + desc_align;

    if ((note_size > shdr->sh_size) ||
        ((shdr->sh_offset + note_size) > file_size) ||
        (note_header->n_namesz == 0) || (note_header->n_descsz == 0)) {
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    name = UCS_PTR_BYTE_OFFSET(note_header, sizeof(Elf32_Nhdr));
    desc = UCS_PTR_BYTE_OFFSET(name, name_align);

    note_out->field_name = ucs_strdup(sec_name, "ucs_elf_note field_name");
    if (note_out->field_name == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }
    note_out->owner = ucs_strndup(name, note_header->n_namesz - 1, "ucs_elf_note owner");
    if (note_out->owner == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_field_name;
    }
    note_out->value_size = note_header->n_descsz;
    note_out->value = ucs_malloc(note_header->n_descsz, "ucs_elf_note value");
    if (note_out->value == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_owner;
    }
    memcpy(note_out->value, desc, note_header->n_descsz);
    return UCS_OK;

err_free_owner:
    ucs_free(note_out->owner);
err_free_field_name:
    ucs_free(note_out->field_name);
err:
    return status;
}

static ucs_status_t ucs_elf_parse_notes(const void *base, size_t file_size,
                                        const char *name_prefix,
                                        ucs_elf_notes_array_t *notes_array)
{
    const Elf64_Ehdr *ehdr = base;
    const Elf64_Shdr *shdr;
    const char *shstrtab;
    Elf64_Word shstrtab_size;
    size_t i;
    ucs_elf_note_t *note;
    const char *sec_name;
    ucs_status_t status;

    status = ucs_elf_get_section_table(base, file_size, &shdr, &shstrtab,
                                       &shstrtab_size);
    if (status != UCS_OK) {
        goto err;
    }

    ucs_array_init_dynamic(notes_array);

    for (i = 0; i < ehdr->e_shnum; i++) {
        if (!ucs_elf_section_matches_prefix(shstrtab,
                                            shstrtab_size,
                                            shdr[i].sh_name,
                                            name_prefix)) {
            continue;
        }

        note     = ucs_array_append(notes_array, 
                                    status = UCS_ERR_NO_MEMORY; goto err);
        sec_name = UCS_PTR_BYTE_OFFSET(shstrtab, shdr[i].sh_name);
        status   = ucs_elf_parse_note(base, file_size, &shdr[i], sec_name, note);
        if (status != UCS_OK) {
            goto err_free_notes;
        }
    }

    return UCS_OK;

err_free_notes:
    ucs_elf_free_notes(notes_array);

err:
    return status;
}

ucs_status_t ucs_elf_read_notes_by_prefix(const char *path,
                                          const char *name_prefix,
                                          ucs_elf_notes_array_t *notes_array)
{
    int fd;
    struct stat st;
    char *map;
    ucs_status_t status;

    if (path == NULL || name_prefix == NULL || notes_array == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    fd  = open(path, O_RDONLY);
    if (fd < 0) {
        ucs_warn("failed to open %s: %m", path);
        return UCS_ERR_IO_ERROR;
    }

    if (fstat(fd, &st) < 0 || (st.st_size < (off_t)sizeof(Elf64_Ehdr))) {
        ucs_warn("failed to stat %s or file too small", path);
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        ucs_warn("mmap(%s) failed: %m", path);
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    if (memcmp(map, ELFMAG, SELFMAG) != 0) {
        ucs_warn("%s is not an ELF file", path);
        status = UCS_ERR_IO_ERROR;
        goto err_munmap;
    }

    if (map[EI_CLASS] != ELFCLASS64) {
        ucs_warn("ELF class 32-bit not supported for note parsing");
        status = UCS_ERR_UNSUPPORTED;
        goto err_munmap;
    }

    status = ucs_elf_parse_notes(map, st.st_size, name_prefix,
                                 notes_array);

err_munmap:
    munmap(map, st.st_size);

out:
    close(fd);
    return status;
}

void ucs_elf_free_notes(ucs_elf_notes_array_t *notes_array)
{
    size_t i;
    ucs_elf_note_t *note;

    if (notes_array == NULL) {
        return;
    }

    for (i = 0; i < ucs_array_length(notes_array); i++) {
        note = &ucs_array_elem(notes_array, i);
        ucs_free(note->value);
        ucs_free(note->owner);
        ucs_free(note->field_name);
    }

    ucs_array_cleanup_dynamic(notes_array);
}

ucs_status_t ucs_elf_read_note_as_string(const ucs_elf_note_t *note,
                                         char *buf, size_t size)
{
    if (note == NULL || buf == NULL || size == 0) {
        return UCS_ERR_INVALID_PARAM;
    }
    if (note->value_size == 0) {
        return UCS_ERR_INVALID_PARAM;
    }
    if (note->value_size >= size) {
        return UCS_ERR_BUFFER_TOO_SMALL;
    }

    memcpy(buf, note->value, note->value_size);
    buf[note->value_size] = '\0';
    return UCS_OK;
}

ucs_status_t ucs_elf_read_note_as_int(const ucs_elf_note_t *note,
                                     int64_t *value_p)
{
    uint32_t v32;

    if (note == NULL || value_p == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    if (note->value_size == 4) {
        memcpy(&v32, note->value, 4);
        *value_p = (int64_t)v32;
        return UCS_OK;
    }
    if (note->value_size == 8) {
        memcpy(value_p, note->value, 8);
        return UCS_OK;
    }

    return UCS_ERR_INVALID_PARAM;
}
