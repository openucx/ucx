/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include <string>
#include <unistd.h>
#include <cstring>

extern "C" {
#include <ucs/sys/elf.h>
#include <ucs/debug/log.h>
}

class test_elf_notes : public ucs::test {
};

const char *const NOTE_PREFIX = ".note.gnu";

UCS_TEST_F(test_elf_notes, read_notes_by_prefix) {
#ifdef GTEST_UCM_HOOK_LIB_DIR
    std::string elf_path = std::string(GTEST_UCM_HOOK_LIB_DIR) + "/libdlopen_test_do_load.so";
    ucs_elf_notes_array_t notes_array;
    ucs_elf_note_t *note;
    ucs_status_t status;

    status = ucs_elf_read_notes_by_prefix(elf_path.c_str(), NOTE_PREFIX, &notes_array);
    ASSERT_EQ(UCS_OK, status);

    /** We expect at least 2 notes: .note.gnu.build-id and .note.gnu.property */
    ASSERT_GE(ucs_array_length(&notes_array), 2);

    ucs_array_for_each(note, &notes_array) {
        ASSERT_EQ(strncmp(note->field_name, NOTE_PREFIX, strlen(NOTE_PREFIX)), 0);
        ASSERT_EQ(strcmp(note->owner, "GNU"), 0);
        ASSERT_TRUE(note->value != NULL);
        EXPECT_GT(note->value_size, 0);
        GTEST_LOG_(INFO) << "note: " << note->field_name << " - owner=" << note->owner << " size=" << note->value_size;
    }
    ucs_elf_free_notes(&notes_array);
#else
    UCS_TEST_SKIP_R("GTEST_UCM_HOOK_LIB_DIR not defined");
#endif
}

UCS_TEST_F(test_elf_notes, free_notes_array_empty) {
    ucs_elf_notes_array_t notes_array;

    ucs_array_init_dynamic(&notes_array);
    ucs_elf_free_notes(&notes_array);
}

UCS_TEST_F(test_elf_notes, read_note_as_string) {
    ucs_elf_note_t note;
    char buf[256];
    ucs_status_t status;
    const char *val = "commit_hash_40_chars_xxxxxxxxxxxxxxxxxx";

    note.value      = (void *)val;
    note.value_size = strlen(val) + 1;

    status = ucs_elf_read_note_as_string(&note, buf, sizeof(buf));
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(std::string(val), std::string(buf));
}

UCS_TEST_F(test_elf_notes, read_note_as_int_4) {
    ucs_elf_note_t note;
    int64_t value = 0;
    ucs_status_t status;
    uint32_t magic = 0x59EC7120;

    note.value      = &magic;
    note.value_size = 4;

    status = ucs_elf_read_note_as_int(&note, &value);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ((int64_t)0x59EC7120, value);
}

UCS_TEST_F(test_elf_notes, read_note_as_int_8) {
    ucs_elf_note_t note;
    int64_t value = 0;
    ucs_status_t status;
    int64_t val8 = (int64_t)0x123456789ABCDEF0ULL;

    note.value      = &val8;
    note.value_size = 8;

    status = ucs_elf_read_note_as_int(&note, &value);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ((int64_t)0x123456789ABCDEF0LL, value);
}

UCS_TEST_F(test_elf_notes, read_note_as_int_invalid_size) {
    ucs_elf_note_t note;
    int64_t value = 0;
    ucs_status_t status;
    char two_bytes[2] = { 0, 0 };

    note.value      = two_bytes;
    note.value_size = 2;

    status = ucs_elf_read_note_as_int(&note, &value);
    EXPECT_NE(UCS_OK, status);
}
