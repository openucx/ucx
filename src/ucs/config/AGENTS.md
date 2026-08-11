# Agent Guide: UCS Configuration Tables

Use these rules for `ucs_config_field_t` arrays and macros that expand to table
elements. They keep definitions and `ucx_info -c` output consistent.

## Element Templates

Place attributes in this order: `name`, `dfl_value`, `doc`, `offset`, `parser`.
Each placeholder below represents a complete C expression.

Set `<offset>` to `ucs_offsetof(<config_type>, <member>)` normally.
Use `0` only when the parser operates on the containing object itself, such as a
subtable at offset zero or a key-value parser whose keys define the offsets.

### User-Visible Field

```c
    {<name>, <dfl_value>,
     <doc>,
     <offset>,
     <parser>},
```

`doc` must follow the rules for documentation in the next section.

### Alias Field (`dfl_value == NULL`)

```c
    {<alias_name>, NULL, "",
     <target_offset>,
     <target_parser>},
```

- Use this form only for a backward-compatible alternate name.
- Use exactly the same offset and parser as the canonical field.
- Place the alias immediately before the canonical field.

### Composition Field

```c
    {<name>, <dfl_value>, NULL,
     <offset>,
     UCS_CONFIG_TYPE_TABLE(<config_table>)},
```

Use this form only to compose/inherit a table or supply its default overrides.

## Documentation

- Keep source lines at or below 80 columns when practical. A modest overrun is
  acceptable to prioritize readability.
- Start sentences with a capital letter and end them with a period.
- End each non-final rendered line with `\n`; do not append `\n` to the final
  line. Never put a space before `\n`.
- Adjacent C string literals do not add whitespace. When wrapping source without
  a rendered line break, end the first literal with one space.
- Do not add blank lines or trailing spaces.
- Put punctuation after macro-expanded text in an adjacent literal, for example
  `UCS_PP_MAKE_STRING(<value>) "."`.
- Document behavior, units, limits, and special values such as `auto` or `inf`.

### Lists

- End text introducing a list with a colon and every list item with a period.
- Indent each item with one space. Align the ` - ` separators in one column,
  one space after the longest item:

```c
     "<list introduction>:\n"
     " <value>      - <description>.\n"
     " <long_value> - <description that wraps onto the next\n"
     "                rendered line>.",
```

- Align wrapped list text with the description after `- ` and use explicit
  `\n` for rendered line breaks.

## Table Layout

- Put one blank line between elements.
- Terminate the table with `{NULL}`.
- Wrap tables, adjacent groups of tables, and macros that expand to table
  elements in `/* clang-format off */` and `/* clang-format on */`.
