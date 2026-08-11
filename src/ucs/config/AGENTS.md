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

### Alias Field

```c
    {<alias_name>, NULL, "",
     <target_offset>,
     <target_parser>},
```

- Use this form only for a backward-compatible alternate name.
- Use exactly the same offset and parser as the canonical field.
- Place the alias immediately before the canonical field so the canonical name
  takes precedence when both names are configured.

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
- Start sentences with a capital letter unless they start with a quoted value.
- End sentences with a period.
- Keep each rendered documentation line in one C string literal. End non-final
  lines with `\n`; do not append `\n` to the final.
- Do not add blank lines or trailing spaces.
- Put punctuation after macro-expanded text in an adjacent literal, for example
  `UCS_PP_MAKE_STRING(<value>) "."`.
- Document behavior, units, limits, and special values such as `auto` or `inf`.
- Wrap literal values mentioned in text outside a list in single quotes, for
  example `'auto'` or `'inf'`, with the exception of comma separated lists of values.

### Lists

- End every list item with a period.
- Text introducing a list may end with a colon or period.
- Start every list-item description with a capital letter.
- Write literal values without quotes. Wrap nonliteral values in `<>`, for
  example `<glob_pattern>` or `file:<filename>`.
- Apply these list rules to key descriptions in `UCS_CONFIG_TYPE_KEY_VALUE` as
  well as directly written documentation strings.
- Apply the same rules recursively to sub-lists. Indent each sub-list item one
  space beyond the start of its parent item's description.
- Align wrapped list text with the description after `- ` and use explicit
  `\n` for rendered line breaks.
- Indent each item with one space. Align the ` - ` separators in one column,
  one space after the longest item.

```c
     " auto           - <Description>.\n"
     " <glob_pattern> - <Description that wraps onto the next\n"
     "                   rendered line>.",
```

## Table Layout

- Put one blank line between elements.
- Terminate the table with `{NULL}`.
- Wrap tables, adjacent groups of tables, and macros that expand to table
  elements in `/* clang-format off */` and `/* clang-format on */`.
