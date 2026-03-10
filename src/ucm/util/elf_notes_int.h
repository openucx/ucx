/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

/**
 * Internal UCM API only. Do not include or use outside UCM.
 * Not installed; not part of the public UCM API.
 */

#ifndef UCM_ELF_NOTES_INT_H_
#define UCM_ELF_NOTES_INT_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/datastruct/array.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>
#include <stddef.h>
#include <stdint.h>

BEGIN_C_DECLS

/**
 * Single ELF note: section name, owner, and descriptor bytes.
 * All pointers are valid until ucm_elf_free_notes().
 */
typedef struct ucm_elf_note {
    char   *field_name;   /**< Full section name (e.g. ".note.nvidia.magic") */
    char   *owner;        /**< Owner name from the note (e.g. "Nvidia") */
    void   *value;        /**< Descriptor bytes (not null-terminated) */
    size_t value_size;    /**< Number of bytes in value */
} ucm_elf_note_t;

/** Dynamic array of ELF notes (UCS array type) */
UCS_ARRAY_DECLARE_TYPE(ucm_elf_notes_array_t, size_t, ucm_elf_note_t);

/**
 * Read all ELF note sections whose name starts with name_prefix into a UCS array.
 * Initializes @a notes_array and fills it; caller must call ucm_elf_free_notes().
 * 64-bit ELF only; 32-bit ELF is not supported.
 *
 * @param [in]  path         Path to the .so file.
 * @param [in]  name_prefix  Section name prefix (e.g. ".note.nvidia").
 * @param [out] notes_array  On success, filled with notes (dynamic array).
 *
 * @return UCS_OK if the file was parsed; zero matches is success.
 *         Error only for open/read/mmap or invalid ELF.
 */
ucs_status_t ucm_elf_read_notes_by_prefix(const char *path,
                                          const char *name_prefix,
                                          ucm_elf_notes_array_t *notes_array);

/**
 * Free a notes array from ucm_elf_read_notes_by_prefix() (frees elements and array).
 * Safe to call with NULL (no-op).
 */
void ucm_elf_free_notes(ucm_elf_notes_array_t *notes_array);

/**
 * Interpret a note's value as a null-terminated string.
 *
 * @param [in]  note  Note (value must be string-like).
 * @param [out] buf   Output buffer.
 * @param [in]  size  Size of buf (including null).
 *
 * @return UCS_OK on success; UCS_ERR_BUFFER_TOO_SMALL or UCS_ERR_INVALID_PARAM.
 */
ucs_status_t ucm_elf_read_note_as_string(const ucm_elf_note_t *note,
                                         char *buf, size_t size);

/**
 * Interpret a note's value as an integer (4- or 8-byte).
 * Integers are in ELF file (host) byte order; cross-endian is not supported.
 *
 * @param [in]  note    Note (value must be 4 or 8 bytes).
 * @param [out] value_p Output value.
 *
 * @return UCS_OK on success; UCS_ERR_INVALID_PARAM if size not 4 or 8.
 */
ucs_status_t ucm_elf_read_note_as_int(const ucm_elf_note_t *note,
                                     int64_t *value_p);

END_C_DECLS

#endif /* UCM_ELF_NOTES_INT_H_ */
