/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/datastruct/string_buffer.h>
#include <ucs/debug/table.h>
}


class test_table : public ucs::test {
protected:
    class table_t {
    public:
        explicit table_t(unsigned n_cols)
        {
            ucs_table_config_t cfg = {
                .n_cols = n_cols
            };
            ucs_table_init(&m_table, &cfg);
        }

        explicit table_t(const ucs_table_config_t &cfg)
        {
            ucs_table_init(&m_table, &cfg);
        }

        ~table_t()
        {
            ucs_table_cleanup(&m_table);
        }

        ucs_table_t *get()
        {
            return &m_table;
        }

        std::string render()
        {
            ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
            ucs_table_render(&m_table, &strb);
            const std::string out(ucs_string_buffer_cstr(&strb));
            ucs_string_buffer_cleanup(&strb);
            return out;
        }

    private:
        ucs_table_t m_table;
    };
};


UCS_TEST_F(test_table, empty_table) {
    table_t table(2);
    EXPECT_EQ("+--+--+\n"
              "+--+--+\n",
              table.render());
}

UCS_TEST_F(test_table, single_cell) {
    table_t table(1);
    auto row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "abc");

    EXPECT_EQ("+-----+\n"
              "| abc |\n"
              "+-----+\n",
              table.render());
}

UCS_TEST_F(test_table, single_cell_empty) {
    table_t table(1);
    auto row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_empty(table.get(), row, 1);

    EXPECT_EQ("+--+\n"
              "|  |\n"
              "+--+\n",
              table.render());
}


UCS_TEST_F(test_table, wide_columns) {
    table_t table(3);

    auto row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "short");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_CENTER,
                               "%s", "even longer cell");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_RIGHT, "%s",
                               "wide third column");

    row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "a much wider value");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_CENTER,
                               "%s", "x");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_RIGHT, "%s",
                               "y");

    EXPECT_EQ("+--------------------+------------------+-------------------+\n"
              "| short              | even longer cell | wide third column |\n"
              "| a much wider value |        x         |                 y |\n"
              "+--------------------+------------------+-------------------+\n",
              table.render());
}

UCS_TEST_F(test_table, separator) {
    table_t table(2);

    auto row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "a");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "b");

    ucs_table_add_separator(table.get());

    row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "c");
    ucs_table_row_add_cell_empty(table.get(), row, 1);

    /* multiple separators */
    ucs_table_add_separator(table.get());
    ucs_table_add_separator(table.get());

    row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_empty(table.get(), row, 1);
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "d");

    EXPECT_EQ("+---+---+\n"
              "| a | b |\n"
              "+---+---+\n"
              "| c |   |\n"
              "+---+---+\n"
              "+---+---+\n"
              "|   | d |\n"
              "+---+---+\n",
              table.render());
}

UCS_TEST_F(test_table, trailing_separator_avoids_bottom_frame) {
    table_t table(2);

    auto row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "x");
    ucs_table_row_add_cell_empty(table.get(), row, 1);

    /* trailing separator, would prevent bottom frame from being rendered */
    ucs_table_add_separator(table.get());

    EXPECT_EQ("+---+--+\n"
              "| x |  |\n"
              "+---+--+\n",
              table.render());
}


UCS_TEST_F(test_table, col_span) {
    table_t table(4);

    auto row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 4, UCS_TABLE_ALIGN_RIGHT,
                               "col_span = %d", 4);

    row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 3, UCS_TABLE_ALIGN_LEFT,
                               "col_span = %d", 3);
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_RIGHT, "%d",
                               1);

    row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 2, UCS_TABLE_ALIGN_CENTER,
                               "left_%d", 2);
    ucs_table_row_add_cell_fmt(table.get(), row, 2, UCS_TABLE_ALIGN_CENTER,
                               "right_%d", 2);

    row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "abcd");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "efgh");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "ijkl");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "mnop");

    EXPECT_EQ("+------+------+------+------+\n"
              "|              col_span = 4 |\n"
              "| col_span = 3       |    1 |\n"
              "|   left_2    |   right_2   |\n"
              "| abcd | efgh | ijkl | mnop |\n"
              "+------+------+------+------+\n",
              table.render());
}

UCS_TEST_F(test_table, col_span_sets_width) {
    table_t table(2);

    auto row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 2, UCS_TABLE_ALIGN_LEFT, "%s",
                               "this header is too wide");

    row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "ab");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_RIGHT, "%s",
                               "cd");

    EXPECT_EQ("+----+--------------------+\n"
              "| this header is too wide |\n"
              "| ab |                 cd |\n"
              "+----+--------------------+\n",
              table.render());
}

UCS_TEST_F(test_table, cell_fmt) {
    table_t table(2);

    auto row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_CENTER,
                               "%d %s..%s", 42, "lo", "hi");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT,
                               "%s=%u", "k", 7u);

    row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT,
                               "%.10f", 3.14159265358979);
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_RIGHT, "%s",
                               "x");

    EXPECT_EQ("+--------------+-----+\n"
              "|  42 lo..hi   | k=7 |\n"
              "| 3.1415926536 |   x |\n"
              "+--------------+-----+\n",
              table.render());
}

UCS_TEST_F(test_table, row_prefix) {
    ucs_table_config_t cfg = {
        .n_cols     = 1,
        .row_prefix = "# ",
    };
    table_t table(cfg);

    auto row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "a");
    ucs_table_add_separator(table.get());
    row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "b");

    EXPECT_EQ("# +---+\n"
              "# | a |\n"
              "# +---+\n"
              "# | b |\n"
              "# +---+\n",
              table.render());
}


UCS_TEST_F(test_table, equal_widths) {
    ucs_table_config_t cfg = {
        .n_cols       = 3,
        .row_prefix   = NULL,
        .equal_widths = 1,
    };
    table_t table(cfg);

    auto row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "a");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "longer");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "xy");

    EXPECT_EQ("+--------+--------+--------+\n"
              "| a      | longer | xy     |\n"
              "+--------+--------+--------+\n",
              table.render());
}

UCS_TEST_F(test_table, equal_widths_with_col_span) {
    ucs_table_config_t cfg = {
        .n_cols       = 2,
        .row_prefix   = NULL,
        .equal_widths = 1,
    };
    table_t table(cfg);

    auto row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 2, UCS_TABLE_ALIGN_LEFT, "%s",
                               "this header is too wide");

    row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "ab");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "cd");

    EXPECT_EQ("+--------------------+--------------------+\n"
              "| this header is too wide                 |\n"
              "| ab                 | cd                 |\n"
              "+--------------------+--------------------+\n",
              table.render());
}


UCS_TEST_F(test_table, render_twice) {
    /* Widths are recomputed on every render: a wider row added between
     * renders must widen the columns on the second render. */
    table_t table(2);
    auto row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "a");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "b");

    EXPECT_EQ("+---+---+\n"
              "| a | b |\n"
              "+---+---+\n",
              table.render());

    row = ucs_table_add_row(table.get());
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "aaaaaaaaaaa");
    ucs_table_row_add_cell_fmt(table.get(), row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               "bbbbbbbbbbb");

    EXPECT_EQ("+-------------+-------------+\n"
              "| a           | b           |\n"
              "| aaaaaaaaaaa | bbbbbbbbbbb |\n"
              "+-------------+-------------+\n",
              table.render());
}
