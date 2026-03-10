/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include <unistd.h>
#include <cstring>

extern "C" {
#include <ucm/util/elf_notes_int.h>
#include <ucs/debug/log.h>
}

class test_elf_notes : public ucs::test {
};

const char *const NOTE_PREFIX = ".note.nvidia";
const char *const VALID_ELF_PATH = "/labhome/rdanino/share/ucx/install/lib/ucx/libuct_ib_plugin.so.0.0.0";

// UCS_TEST_F(test_elf_notes, invalid_path) {
//     ucm_elf_notes_array_t notes_array = UCS_ARRAY_STATIC_INITIALIZER;
//     ucs_status_t status;

//     status = ucm_elf_read_notes_by_prefix("/nonexistent/path/lib.so",
//                                           NOTE_PREFIX, &notes_array);
//     EXPECT_EQ(UCS_ERR_IO_ERROR, status);
//     EXPECT_TRUE(ucs_array_is_empty(&notes_array));
//     ucm_elf_free_notes(&notes_array);
// }

UCS_TEST_F(test_elf_notes, valid_elf_zero_notes) {
    ucm_elf_notes_array_t notes_array;
    ucm_elf_note_t *note;
    ucs_status_t status;


    status = ucm_elf_read_notes_by_prefix(VALID_ELF_PATH, ".note.nvidia", &notes_array);
    EXPECT_EQ(UCS_OK, status);
    ucs_array_for_each(note, &notes_array) {
        ucs_info("note: %s - owner=%s size=%zu", note->field_name, note->owner, note->value_size);
        if (note->value_size == 4 || note->value_size == 8) {
            int64_t value = 0;
            status = ucm_elf_read_note_as_int(note, &value);
            EXPECT_EQ(UCS_OK, status);
            ucs_info("value: 0x%lx", value);
        } else {
            char buf[256];
            status = ucm_elf_read_note_as_string(note, buf, sizeof(buf));
            EXPECT_EQ(UCS_OK, status);
            ucs_info("value: %s", buf);
        }
    }
    ucm_elf_free_notes(&notes_array);
}

UCS_TEST_F(test_elf_notes, free_notes_array_empty) {
    ucm_elf_notes_array_t notes_array;

    ucs_array_init_dynamic(&notes_array);
    ucm_elf_free_notes(&notes_array);
}

UCS_TEST_F(test_elf_notes, read_note_as_string) {
    ucm_elf_note_t note;
    char buf[256];
    ucs_status_t status;
    const char *val = "commit_hash_40_chars_xxxxxxxxxxxxxxxxxx";

    note.value      = (void *)val;
    note.value_size = strlen(val) + 1;

    status = ucm_elf_read_note_as_string(&note, buf, sizeof(buf));
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(std::string(val), std::string(buf));
}

UCS_TEST_F(test_elf_notes, read_note_as_int_4) {
    ucm_elf_note_t note;
    int64_t value = 0;
    ucs_status_t status;
    uint32_t magic = 0x59EC7120;

    note.value      = &magic;
    note.value_size = 4;

    status = ucm_elf_read_note_as_int(&note, &value);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ((int64_t)0x59EC7120, value);
}

UCS_TEST_F(test_elf_notes, read_note_as_int_8) {
    ucm_elf_note_t note;
    int64_t value = 0;
    ucs_status_t status;
    int64_t val8 = (int64_t)0x123456789ABCDEF0ULL;

    note.value      = &val8;
    note.value_size = 8;

    status = ucm_elf_read_note_as_int(&note, &value);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ((int64_t)0x123456789ABCDEF0LL, value);
}

UCS_TEST_F(test_elf_notes, read_note_as_int_invalid_size) {
    ucm_elf_note_t note;
    int64_t value = 0;
    ucs_status_t status;
    char two_bytes[2] = { 0, 0 };

    note.value      = two_bytes;
    note.value_size = 2;

    status = ucm_elf_read_note_as_int(&note, &value);
    EXPECT_NE(UCS_OK, status);
}
